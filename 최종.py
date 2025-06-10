# pip install Pillow
# pip install opencv-python
# pip install numpy
# pip install torch
# pip install torchvision

import tkinter as tk
from tkinter import filedialog, ttk, Scale, Label, Button, Frame, Radiobutton, StringVar, Canvas, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# PyTorch 및 관련 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout)

    def build_conv_block(self, dim, norm_layer, use_dropout):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# --- 필터 세기 조절---
def map_value(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(value, in_max))
    if (in_max - in_min) == 0:
        return out_min if out_min < out_max else (out_min + out_max) / 2 
    mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return mapped

# --- 카툰 효과 필터 ---
def Cartoon_effect(img_cv, strength=50):
    if img_cv is None: return None
    k = int(map_value(strength, 0, 100, 20, 3))
    canny_thresh1 = int(map_value(strength, 0, 100, 100, 30))
    canny_thresh2 = int(map_value(strength, 0, 100, 200, 70))
    k_idx = int(map_value(strength, 0, 100, 0, 4.99))
    median_ksize = 3 + k_idx * 2

    data = np.float32(img_cv).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quantized_img = center[label.flatten()]
    quantized_img = quantized_img.reshape(img_cv.shape)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, median_ksize)
    edges = cv2.Canny(gray_blurred, canny_thresh1, canny_thresh2)
    
    cartoon_img = quantized_img.copy()
    cartoon_img[edges == 255] = [0, 0, 0]
    return cartoon_img

# --- 연필 드로잉 필터 ---
def PencilSketch_effect(img, strength=50):
    if img is None: return None
    blur_ksize = int(map_value(strength, 0, 100, 3, 51))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray_img = 255 - gray_img
    blurred_img = cv2.GaussianBlur(inverted_gray_img, (blur_ksize, blur_ksize), 0)

    divisor = np.maximum(255 - blurred_img, 1)
    sketch_img = cv2.divide(gray_img.astype(np.float32), divisor.astype(np.float32), scale=256.0)
    sketch_img = np.uint8(np.clip(sketch_img, 0, 255))

    return sketch_img

# --- 하프톤 필터 (4x4 베이어 행렬) ---
def Halftone_effect(img_cv, strength=50):
    if img_cv is None: return None
    scale_factor_percent = int(map_value(strength, 0, 100, 100, 20))
    scale_factor = scale_factor_percent / 100.0

    if scale_factor < 0.1:
        scale_factor = 0.1

    img_resized = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    bayer_matrix_4x4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]])
    
    m_size = bayer_matrix_4x4.shape[0]
    levels = m_size * m_size

    halftone_img = np.zeros_like(gray_img, dtype=np.uint8)
    
    for r in range(gray_img.shape[0]):
        for c in range(gray_img.shape[1]):
            threshold_from_bayer = bayer_matrix_4x4[r % m_size, c % m_size]
            pixel_val_normalized = int(gray_img[r,c] / (256.0 / levels))
            
            if pixel_val_normalized > threshold_from_bayer :
                halftone_img[r, c] = 255
            else:
                halftone_img[r, c] = 0
    return halftone_img

# --- GAN 인상주의 필터 ---
def GAN_Impressionist_effect(img_cv_original, gan_model, transform, device, strength=100):
    if img_cv_original is None or gan_model is None: return None

    if len(img_cv_original.shape) == 2:
        img_cv_original = cv2.cvtColor(img_cv_original, cv2.COLOR_GRAY2BGR)

    img_pil_input = Image.fromarray(cv2.cvtColor(img_cv_original, cv2.COLOR_BGR2RGB))
    
    img_tensor = transform(img_pil_input).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = gan_model(img_tensor)

    output_img_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img_np = (output_img_np * 0.5 + 0.5) * 255
    output_img_np = np.uint8(np.clip(output_img_np, 0, 255))
    gan_output_cv = cv2.cvtColor(output_img_np, cv2.COLOR_RGB2BGR)
    
    gan_output_resized = cv2.resize(gan_output_cv, 
                                     (img_cv_original.shape[1], img_cv_original.shape[0]), 
                                     interpolation=cv2.INTER_LANCZOS4)

    alpha = strength / 100.0
    blended_img = cv2.addWeighted(gan_output_resized, alpha, img_cv_original, 1 - alpha, 0)
    
    return blended_img


# --- 이미지 필터 애플리케이션 클래스 ---
class ImageFilterApp:
    MAX_INDIVIDUAL_DISPLAY_WIDTH = 500
    MAX_INDIVIDUAL_DISPLAY_HEIGHT = 450
    CONTROLS_WIDTH = 220

    def __init__(self, master):
        self.master = master
        master.title("이미지 필터 및 GAN 스타일 변환")

        self.original_cv_image = None
        self.processed_cv_image = None
        self.last_loaded_image_size = (0,0)
        self.current_filter_strength = tk.IntVar(value=50)

        self.gan_model_G_B = None
        self.gan_transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 또는 CPU 디바이스 설정
        self.load_gan_model()

        main_frame = Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = Frame(main_frame, width=self.CONTROLS_WIDTH, bd=2, relief=tk.GROOVE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        self.image_display_frame = Frame(main_frame)
        self.image_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        load_button = Button(control_frame, text="이미지 로드", command=self.load_image)
        load_button.pack(pady=5, fill=tk.X, padx=5)

        self.save_button = Button(control_frame, text="이미지 저장", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=5, fill=tk.X, padx=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)

        Label(control_frame, text="필터 선택:").pack(pady=5, anchor=tk.W, padx=5)
        self.filter_var = StringVar(value="Cartoon")
        filters = [("카툰 효과", "Cartoon"), 
                   ("연필 드로잉", "PencilSketch"), 
                   ("하프톤 (4x4)", "Halftone4x4"),
                   ("GAN 인상주의", "GANImpressionist")]
        for text, value in filters:
            Radiobutton(control_frame, text=text, variable=self.filter_var, value=value, 
                        command=self.on_filter_selected).pack(anchor=tk.W, padx=15)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10, padx=5)

        Label(control_frame, text="필터 세기:").pack(pady=5, anchor=tk.W, padx=5)
        self.strength_slider = Scale(control_frame, variable=self.current_filter_strength, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, length=180, command=self.on_strength_changed)
        self.strength_slider.pack(fill=tk.X, padx=15)
        
        self.apply_button = Button(control_frame, text="필터 적용", command=self.apply_filter, state=tk.DISABLED)
        self.apply_button.pack(pady=10, fill=tk.X, padx=5)

        self.original_image_label = Label(self.image_display_frame, text="원본 이미지", relief=tk.SUNKEN)
        self.original_image_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.processed_image_label = Label(self.image_display_frame, text="처리된 이미지", relief=tk.SUNKEN)
        self.processed_image_label.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        master.geometry(f"{self.CONTROLS_WIDTH + 2 * 150 + 40}x{self.MAX_INDIVIDUAL_DISPLAY_HEIGHT + 40}")
        master.minsize(self.CONTROLS_WIDTH + 2*50 + 30, 200)

    # --- GAN 모델 로드 메서드 ---
    def load_gan_model(self):
        gan_model_path = r'C:\Users\jsi01\OneDrive\바탕 화면\컴퓨터비전기반오토모티브SW\latest_net_G_B.pth' 

        try:
            self.gan_model_G_B = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
            
            state_dict = torch.load(gan_model_path, map_location=self.device)
            self.gan_model_G_B.load_state_dict(state_dict)
            self.gan_model_G_B.to(self.device)
            self.gan_model_G_B.eval()

            self.gan_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            print(f"GAN 모델 '{gan_model_path}'이(가) 성공적으로 로드되었습니다.")
            print(f"GAN 모델이 사용하는 디바이스: {self.device}")
        except FileNotFoundError:
            messagebox.showwarning("GAN 모델 로드 실패", 
                                   f"GAN 모델 파일 '{gan_model_path}'을(를) 찾을 수 없습니다.\n"
                                   "경로를 확인하거나, 모델 파일을 실행 파일과 같은 디렉토리에 두세요.")
            self.gan_model_G_B = None
        except Exception as e:
            messagebox.showerror("GAN 모델 로드 오류", 
                                 f"GAN 모델 로드 중 오류 발생: {e}\n"
                                 "Generator 클래스 정의가 학습된 모델과 일치하는지 확인하세요.")
            self.gan_model_G_B = None

    # --- 필터 선택 시 호출되는 메서드 ---
    def on_filter_selected(self):
        pass
    
    # --- 필터 세기 슬라이더 조작 시 호출되는 메서드 ---
    def on_strength_changed(self, value):
        if self.original_cv_image is not None and self.filter_var.get() == "GANImpressionist":
             self.apply_filter()
        else:
            if self.original_cv_image is not None and self.filter_var.get() != "GANImpressionist":
                self.apply_filter()


    # --- 이미지 표시 크기 계산 메서드 ---
    def _get_display_size(self, img_w, img_h):
        scale_w = self.MAX_INDIVIDUAL_DISPLAY_WIDTH / img_w if img_w > 0 else 1
        scale_h = self.MAX_INDIVIDUAL_DISPLAY_HEIGHT / img_h if img_h > 0 else 1
        scale = min(scale_w, scale_h, 1.0))
        
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        return display_w, display_h

    # --- OpenCV 이미지를 Tkinter 표시용 이미지로 변환하는 메서드 ---
    def _cv_to_tk(self, cv_image, display_w, display_h):
        if cv_image is None: return None
        resized_cv_image = cv2.resize(cv_image, (display_w, display_h), interpolation=cv2.INTER_AREA)

        if len(resized_cv_image.shape) == 2:
            img_rgb = cv2.cvtColor(resized_cv_image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(resized_cv_image, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(image=img_pil)

    # --- 윈도우 크기 조절 메서드 ---
    def _adjust_window_size(self):
        if self.original_cv_image is None:
            self.master.geometry(f"{self.CONTROLS_WIDTH + 2 * 150 + 40}x{self.MAX_INDIVIDUAL_DISPLAY_HEIGHT + 40}")
            return

        orig_h, orig_w = self.original_cv_image.shape[:2]
        ideal_display_w, ideal_display_h = self._get_display_size(orig_w, orig_h)
        
        total_width = self.CONTROLS_WIDTH + (2 * ideal_display_w) + 40
        total_height = ideal_display_h + 40

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        total_width = min(total_width, screen_width - 50)
        total_height = min(total_height, screen_height - 50)

        self.master.geometry(f"{total_width}x{total_height}")

    # --- 이미지를 Tkinter Label에 표시하는 메서드 ---
    def display_image_on_label(self, cv_img, label_widget, is_original=False):
        if cv_img is None:
            label_widget.config(image='', text="이미지 없음")
            if hasattr(label_widget, 'image_ref'):
                del label_widget.image_ref
            return

        # 표시 크기 결정 (원본 이미지 크기 기준)
        if self.original_cv_image is not None:
            orig_h, orig_w = self.original_cv_image.shape[:2]
            disp_w, disp_h = self._get_display_size(orig_w, orig_h)
        elif cv_img is not None:
            proc_h, proc_w = cv_img.shape[:2]
            disp_w, disp_h = self._get_display_size(proc_w, proc_h)
        else:
            disp_w, disp_h = 150, 150

        img_tk = self._cv_to_tk(cv_img, disp_w, disp_h)
        label_widget.config(image=img_tk, text="")
        label_widget.image_ref = img_tk

        if is_original:
            self._adjust_window_size()

    # --- 이미지 로드 버튼 클릭 시 호출되는 메서드 ---
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=(("이미지 파일", "*.jpg *.jpeg *.png"), ("모든 파일", "*.*"))
        )
        if file_path:
            try:
                self.original_cv_image = cv2.imread(file_path)
                if self.original_cv_image is None:
                    raise ValueError("OpenCV가 이미지를 로드하지 못했습니다. 파일이 유효한 이미지 파일인지 확인하세요.")
                
                self.last_loaded_image_size = (self.original_cv_image.shape[1], self.original_cv_image.shape[0])
                self.display_image_on_label(self.original_cv_image, self.original_image_label, is_original=True)
                
                self.processed_cv_image = None
                self.display_image_on_label(self.processed_cv_image, self.processed_image_label)
                
                self.apply_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("오류", f"이미지 로드 실패: {e}\n파일 경로: {file_path}")
                self.original_cv_image = None
                self.display_image_on_label(None, self.original_image_label, is_original=True)

    # --- 이미지 저장 버튼 클릭 시 호출되는 메서드 ---
    def save_image(self):
        if self.processed_cv_image is None:
            messagebox.showwarning("경고", "저장할 처리된 이미지가 없습니다.")
            return
        file_path = filedialog.asksaveasfilename(
            title="처리된 이미지 저장",
            defaultextension=".png",
            filetypes=(("PNG 파일", "*.png"), ("JPEG 파일", "*.jpg"))
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_cv_image)
                messagebox.showinfo("성공", f"이미지가 저장되었습니다: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"이미지 저장 중 오류 발생: {e}")

    # --- 필터 적용 버튼 클릭 시 호출되는 메서드 ---
    def apply_filter(self):
        if self.original_cv_image is None:
            messagebox.showwarning("경고", "먼저 이미지를 로드하세요.")
            return

        filter_type = self.filter_var.get()
        strength = self.current_filter_strength.get()
        img_to_process = self.original_cv_image.copy()

        try:
            if filter_type == "Cartoon":
                self.processed_cv_image = Cartoon_effect(img_to_process, strength=strength)
            elif filter_type == "PencilSketch":
                self.processed_cv_image = PencilSketch_effect(img_to_process, strength=strength)
            elif filter_type == "Halftone4x4":
                self.processed_cv_image = Halftone_effect(img_to_process, strength=strength)
            elif filter_type == "GANImpressionist":
                if self.gan_model_G_B and self.gan_transform:
                    self.processed_cv_image = GAN_Impressionist_effect(
                        img_to_process, self.gan_model_G_B, self.gan_transform, self.device, strength=strength
                    )
                else:
                    messagebox.showwarning("GAN 모델 없음", "GAN 모델이 로드되지 않았거나 변환 준비가 되지 않았습니다.")
                    self.processed_cv_image = None
            else: # 필터가 선택되지 않았거나 알 수 없는 필터일 경우
                self.processed_cv_image = img_to_process 
        except Exception as e:
            messagebox.showerror("필터 적용 오류", f"필터 적용 중 오류 발생: {e}")
            self.processed_cv_image = None

        if self.processed_cv_image is not None:
            self.display_image_on_label(self.processed_cv_image, self.processed_image_label)
            self.save_button.config(state=tk.NORMAL)
        else:
            self.display_image_on_label(None, self.processed_image_label)
            self.save_button.config(state=tk.DISABLED)

# --- 애플리케이션 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()
