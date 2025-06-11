# 만화 스타일로 이미지 변환 - Converting-images
Converting images to cartoon style 
[![Video Label](http://img.youtube.com/vi/WO7ePyKYrlo/0.jpg)](https://youtu.be/WO7ePyKYrlo)

🎨 이미지 필터 및 GAN 스타일 변환 앱 🖼️

프로젝트 개요

이 애플리케이션은 Python의 Tkinter를 기반으로 한 직관적인 GUI를 통해 이미지에 다양한 시각 효과를 적용할 수 있는 도구입니다. 기본적인 필터(카툰, 연필 드로잉, 하프톤)뿐만 아니라, 학습된 GAN(Generative Adversarial Network) 모델을 활용하여 일반 사진을 인상주의 회화 스타일로 변환하는 혁신적인 기능을 제공합니다. 사용자 친화적인 인터페이스로 누구나 쉽게 이미지를 예술 작품으로 탈바꿈시킬 수 있습니다.

✨ 주요 기능
이미지 로드 및 저장: JPG, PNG 등 다양한 형식의 이미지 파일을 손쉽게 불러오고 처리된 이미지를 저장합니다.
카툰 효과: 이미지의 색상을 단순화하고 윤곽선을 강조하여 만화 같은 느낌을 줍니다.
연필 드로잉: 이미지를 흑백 연필 스케치처럼 변환하여 섬세한 그림 효과를 표현합니다.
하프톤 효과 (4x4 베이어): 이미지를 신문 인쇄물처럼 점(dot) 패턴으로 분해하여 독특한 질감을 만듭니다.
GAN 인상주의 변환: 학습된 CycleGAN 모델을 사용하여 일반 사진을 인상주의 회화 스타일로 자동 변환합니다.강도 조절: 원본 이미지와 GAN 변환 결과물을 블렌딩하여 인상주의 효과의 강도를 조절할 수 있습니다.

🛠️ 기술 스택
Python 3.xTkinter: GUI (Graphical User Interface) 구축OpenCV (cv2): 이미지 처리 및 조작Pillow (PIL): 이미지 파일 관리 및 Tkinter 호환성NumPy: 고성능 수치 계산 및 이미지 데이터 처리PyTorch: GAN 모델 정의, 학습, 추론 (CPU 또는 CUDA)

🚀 설치 및 실행 방법

1. 필수 라이브러리 설치터미널 또는 명령 프롬프트에서 다음 명령어를 실행하여 필요한 라이브러리들을 설치합니다.pip install Pillow
pip install opencv-python
pip install numpy
# PyTorch 설치 (사용하는 환경에 따라 적절한 명령어 선택)
# CPU만 사용:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# NVIDIA GPU (CUDA 12.1):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
2. GAN 모델 파일 준비학습을 통해 얻은 GAN 모델의 가중치 파일(예: latest_net_G_B.pth)을 준비합니다.이 파일을 앱의 Python 스크립트(image_filter_app.py)와 동일한 디렉토리에 놓는 것을 권장합니다.만약 다른 경로에 있다면, 코드 내 load_gan_model 함수의 gan_model_path 변수를 해당 파일의 정확한 경로로 수정해야 합니다.# 예시:
gan_model_path = r'C:\Users\jsi01\OneDrive\바탕 화면\컴퓨터비전기반오토모티브SW\latest_net_G_B.pth'
(r 접두사는 Windows 경로의 역슬래시 문제를 해결해 줍니다.)3. 애플리케이션 실행Python 스크립트를 실행합니다.python image_filter_app.py

📝 사용 방법
이미지 로드 버튼을 클릭하여 원하는 이미지를 불러옵니다.왼쪽 패널의 필터 선택 섹션에서 원하는 필터(카툰 효과, 연필 드로잉, 하프톤, GAN 인상주의)를 선택합니다.필터 세기 슬라이더를 조절하여 효과의 강도를 변경합니다. (GAN 인상주의 필터의 경우, 이 슬라이더가 원본과 변환 결과의 혼합 비율을 조절합니다.)필터 적용 버튼을 클릭하여 선택된 필터를 이미지에 적용합니다.처리된 이미지는 오른쪽 패널에 표시됩니다.이미지 저장 버튼을 클릭하여 결과물을 저장할 수 있습니다.

💡 개발 과정: 문제점 및 해결 과정이 애플리케이션 개발은 기본적인 이미지 처리부터 딥러닝 모델 통합까지 다층적인 도전을 포함했습니다.

1. 초기 이미지 필터 앱 구현문제점: Tkinter의 제한된 이미지 처리 능력과 OpenCV의 강력한 기능을 연동해야 했습니다. 특히 NumPy 배열 형태의 OpenCV 이미지를 Tkinter GUI에 직접 표시하기 어려웠습니다.해결: Pillow (PIL) 라이브러리를 중간 다리 역할로 활용했습니다. OpenCV 이미지를 PIL 이미지로, 다시 Tkinter의 PhotoImage 객체로 변환하는 파이프라인을 구축하여 GUI에 원활하게 이미지를 표시할 수 있었습니다. 또한, _get_display_size 및 _adjust_window_size 함수를 통해 다양한 크기의 이미지를 로드할 때 GUI 창 크기가 동적으로 조절되도록 하여 사용자 경험을 개선했습니다. 각 필터의 세기 조절은 map_value 함수를 통해 슬라이더 값을 필터 파라미터에 선형적으로 매핑하여 구현했습니다.

2. GAN 모델 학습 (Colab 환경)문제점: 고품질의 인상주의 스타일 변환을 위해 CycleGAN 모델을 학습해야 했으나, 대규모의 이미지 데이터셋(원본 사진 및 인상주의 그림)을 Colab 런타임에 효율적으로 불러오고 관리하는 것이 어려웠습니다. 또한, 학습은 많은 시간과 GPU 자원을 요구하여 Colab 런타임 제한으로 인해 학습이 자주 중단되는 문제가 발생했습니다.해결:Google Drive 마운트: 데이터셋을 Google Drive에 저장하고 Colab 노트북에 마운트하여 런타임 재시작 시에도 데이터를 지속적으로 사용할 수 있도록 했습니다.체크포인트 저장: 학습 과정 중 모델의 가중치(state_dict) 및 옵티마이저 상태를 주기적으로 Google Drive에 .pth 파일 형태로 저장하는 체크포인트 기능을 구현했습니다. 이를 통해 런타임이 중단되더라도 최신 저장 지점부터 학습을 재개할 수 있어 효율적인 개발이 가능했습니다.데이터 전처리: CycleGAN 학습에 맞춰 이미지들을 고정된 크기(예: 256x256)로 리사이즈하고 PyTorch 텐서로 변환하며, 픽셀 값을 [-1, 1] 범위로 정규화하는 전처리 파이프라인을 torchvision.transforms를 사용하여 구축했습니다.

3. GAN 모델 통합문제점: 학습된 PyTorch GAN 모델을 Tkinter 앱에 연동하고, OpenCV 이미지 형식과 PyTorch 텐서 형식 간의 변환을 원활하게 처리해야 했습니다. 또한, GAN 모델은 직접적인 '필터 세기' 파라미터를 제공하지 않아 사용자에게 익숙한 세기 조절 기능을 제공하기 어려웠습니다.해결:load_gan_model 메서드를 통해 앱 시작 시 GAN 모델의 가중치를 로드하고, 시스템의 GPU 사용 가능 여부(CUDA)를 자동으로 감지하여 모델을 최적의 디바이스로 이동시켰습니다.GAN_Impressionist_effect 함수를 구현하여 OpenCV 이미지를 PyTorch 텐서로 전처리 후 GAN 모델을 통해 스타일 변환을 수행하고, 다시 OpenCV 이미지로 후처리하는 과정을 자동화했습니다.GAN 필터의 세기 조절은 알파 블렌딩 기법을 사용하여 구현했습니다. cv2.addWeighted() 함수를 통해 원본 이미지와 GAN 변환 결과 이미지를 슬라이더 값에 따라 비율적으로 혼합함으로써, 사용자가 인상주의 효과의 강도를 직관적으로 조절할 수 있도록 했습니다.

⚙️ 주요 함수 및 알고리즘 적용

애플리케이션에 구현된 각 필터 함수와 그 안에 적용된 핵심 알고리즘은 다음과 같습니다.map_value(value, in_min, in_max, out_min, out_max)설명: 입력된 value를 in_min부터 in_max까지의 범위에서 out_min부터 out_max까지의 새로운 범위로 선형적으로 매핑하는 유틸리티 함수입니다. 필터 세기 슬라이더(0~100)의 값을 각 필터의 고유한 파라미터 범위로 변환하는 데 사용됩니다.
알고리즘: 선형 보간(Linear Interpolation)Cartoon_effect(img_cv, strength)설명: 이미지를 만화처럼 보이게 색상을 단순화하고 윤곽선을 강조하는 필터입니다.
알고리즘: **K-평균 군집화 (K-Means Clustering)**를 통한 색상 양자화, **중앙값 필터 (Median Blur)**를 통한 노이즈 제거, **캐니 엣지 검출 (Canny Edge Detection)**을 통한 윤곽선 강조, 그리고 이 두 결과를 합성합니다.
PencilSketch_effect(img, strength)설명: 이미지를 연필 드로잉처럼 흑백 음영으로 변환합니다.알고리즘: 이미지 반전, 가우시안 블러 (Gaussian Blur), 그리고 색상 닷지 혼합 (Color Dodge Blend) 효과를 구현하여 연필 스케치의 질감을 표현합니다.
Halftone_effect(img_cv, strength)설명: 이미지를 신문 인쇄물처럼 점(dot) 패턴으로 표현합니다.알고리즘: 이미지 리사이즈를 통한 망점 크기 조절과 **순서형 디더링 (Ordered Dithering) with 베이어 행렬 (Bayer Matrix)**을 사용하여 밝기를 망점의 밀도로 나타냅니다.
GAN_Impressionist_effect(img_cv_original, gan_model, transform, device, strength)설명: 학습된 GAN 모델을 사용하여 원본 사진을 인상주의 그림 스타일로 변환하며, '필터 세기'는 원본과 변환 결과의 혼합 비율을 조절합니다.알고리즘: **GAN 추론 (GAN Inference)**을 통해 스타일 변환을 수행하고, **알파 블렌딩 (Alpha Blending)**을 통해 원본 이미지와 GAN 결과 이미지를 혼합하여 효과의 강도를 조절합니다.

✅ 결론 및 향후 과제

결론
본 애플리케이션은 Tkinter, OpenCV, PyTorch를 성공적으로 통합하여 사용자에게 다채로운 이미지 필터링 경험을 제공합니다. 특히 GAN 모델의 연동은 단순한 필터링을 넘어, 딥러닝 기반의 예술적 스타일 변환 기술이 실제 사용자 앱에서 어떻게 구현될 수 있는지 보여주는 중요한 사례가 되었습니다. 사용자는 직관적인 GUI를 통해 이미지의 새로운 예술적 가능성을 탐색할 수 있습니다.향후 과제현재 구현된 애플리케이션의 성능과 기능을 더욱 향상시키기 위한 몇 가지 향후 과제는 다음과 같습니다.성능 최적화:GAN 모델의 추론 속도 개선: 특히 CPU 환경에서의 지연을 줄이기 위해 모델 경량화(quantization, pruning)나 ONNX/TensorRT 같은 추론 최적화 프레임워크 도입을 고려할 수 있습니다.실시간 미리보기: 필터 세기 조절 시 GAN 필터도 즉시 미리보기를 제공할 수 있도록 비동기 처리나 쓰레딩을 적용하여 GUI 응답성을 높이는 것이 필요합니다.확장성 및 유연성:다양한 GAN 모델 지원: 사용자가 다른 GAN 모델(예: 특정 화가 스타일 모델, 다른 스타일 변환 모델)을 선택하여 로드할 수 있도록 모델 선택 기능을 추가할 수 있습니다.사용자 정의 변환: 단순히 스타일을 변경하는 것을 넘어, 사용자가 특정 컨텐츠를 기반으로 새로운 요소를 생성하거나 변형하는 등의 고급 GAN 기능을 탐색할 수 있도록 확장할 수 있습니다.UI/UX 개선:진행률 표시: 필터 적용 시 사용자에게 작업이 진행 중임을 알리는 진행률 바 또는 스피너를 추가하여 사용자 경험을 향상시킬 수 있습니다.이미지 미리보기 확대/축소 및 이동 기능: 큰 이미지를 세밀하게 확인하거나 특정 영역을 집중적으로 볼 수 있는 기능을 추가하면 좋습니다.추가 필터 및 기능:사진 수정 도구: 밝기, 대비, 채도 등 기본적인 사진 보정 도구를 추가하여 애플리케이션의 유용성을 높일 수 있습니다.배치 처리: 여러 이미지를 한 번에 선택하여 동일한 필터를 적용하고 저장하는 기능을 추가할 수 있습니다.이러한 향후 과제들을 해결해나가면서 애플리케이션은 더욱 강력하고 사용자 친화적인 이미지 처리 도구로 발전할 수 있을 것입니다.
