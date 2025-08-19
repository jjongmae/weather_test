import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# 1. Hugging Face에서 사전 학습된 모델 불러오기
#    최초 실행 시 모델 파일을 자동으로 다운로드합니다. (수 분 소요될 수 있음)
pipe = pipeline("image-classification", model="prithivMLmods/Weather-Image-Classification")

# cloudy/overcast (흐림)
# foggy/hazy (안개)
# rain/storm (비)
# snow/frosty (눈)
# sun/clear (맑음)

# 2. 분석할 이미지 파일 경로 지정
#    'my_cctv_image.jpg' 부분을 실제 파일명으로 변경하세요.
#    인터넷 상의 이미지 주소를 넣어도 됩니다.
image_path = "image/fog_2.png" 

try:
    # 3. 이미지 열기 (RGBA 모드로 변환하여 투명도 처리)
    image = Image.open(image_path).convert("RGBA")

    # 4. 모델을 사용하여 이미지 분류 실행
    results = pipe(image)

    # 5. 분석 결과를 이미지에 표시하기
    # 결과를 텍스트로 변환
    text_to_draw = "--- 분석 결과 ---\n"
    for result in results:
        label = result['label']
        score = result['score']
        text_to_draw += f"날씨: {label}, 신뢰도: {score:.2%}\n"

    # 이미지에 그리기를 위한 Draw 객체 생성
    draw = ImageDraw.Draw(image)
    
    # 한글 출력을 위한 폰트 설정 (Windows의 '맑은 고딕' 폰트 사용)
    try:
        font = ImageFont.truetype("malgun.ttf", size=20)
    except IOError:
        print("맑은 고딕 폰트(malgun.ttf)를 찾을 수 없습니다. 한글이 깨질 수 있습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 텍스트 크기 및 위치 계산
    text_bbox = draw.textbbox((0, 0), text_to_draw, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width, image_height = image.size
    
    # 우측 상단 위치 (여백 10px)
    position = (image_width - text_width - 10, 10)

    # 텍스트 가독성을 위한 반투명 배경 추가
    background_position = (
        position[0] - 5, 
        position[1] - 5, 
        position[0] + text_width + 5, 
        position[1] + text_height + 5
    )
    draw.rectangle(background_position, fill=(0, 0, 0, 128)) # 반투명 검은색

    # 텍스트 그리기 (흰색)
    draw.text(position, text_to_draw, font=font, fill=(255, 255, 255))

    # 6. 수정된 이미지 보기
    image.show()

    # 7. 콘솔에도 결과 출력
    print("--- 분석 결과 ---")
    for result in results:
        label = result['label']
        score = result['score']
        print(f"날씨: {label}, 신뢰도: {score:.2%}")


except FileNotFoundError:
    print(f"오류: '{image_path}' 파일을 찾을 수 없습니다. 파일명과 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
