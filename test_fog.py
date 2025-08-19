import cv2
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Dark Channel Prior (DCP) 기반 안개 제거 함수
def get_dark_channel(img, patch_size):
    M, N, _ = img.shape
    padded = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), mode='edge')
    dark_channel = np.zeros((M, N))
    for i, j in np.ndindex(M, N):
        dark_channel[i, j] = np.min(padded[i:i + patch_size, j:j + patch_size, :])
    return dark_channel

def estimate_atmospheric_light(img, dark_channel, top_pixels=0.001):
    h, w, _ = img.shape
    flat_img = img.reshape(h * w, 3)
    flat_dark_channel = dark_channel.ravel()
    
    # 어두운 채널 값이 가장 높은 상위 0.1% 픽셀 선택
    num_pixels = int(h * w * top_pixels)
    indices = np.argsort(flat_dark_channel)[-num_pixels:]
    
    # 해당 픽셀들 중에서 가장 밝은 픽셀의 값을 대기광으로 추정
    atmospheric_light = np.max(flat_img[indices], axis=0)
    return atmospheric_light

def estimate_transmission(img, atmospheric_light, omega=0.95, patch_size=15):
    normalized_img = img / atmospheric_light
    transmission = 1 - omega * get_dark_channel(normalized_img, patch_size)
    return transmission

def recover_scene_radiance(img, transmission, atmospheric_light, t0=0.1):
    transmission = np.maximum(transmission, t0)
    scene_radiance = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        scene_radiance[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    return np.clip(scene_radiance, 0, 255).astype(np.uint8)

def dehaze_image_dcp(hazy_img_bgr):
    # BGR 이미지를 RGB로 변환 (DCP는 RGB에서 작동)
    hazy_img_rgb = cv2.cvtColor(hazy_img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Dark Channel 계산
    dark_channel = get_dark_channel(hazy_img_rgb, patch_size=15)
    
    # 2. 대기광 추정
    atmospheric_light = estimate_atmospheric_light(hazy_img_rgb, dark_channel)
    
    # 3. 투과율 맵 추정
    transmission = estimate_transmission(hazy_img_rgb, atmospheric_light, patch_size=15)
    
    # 4. 장면 복원
    dehazed_img_rgb = recover_scene_radiance(hazy_img_rgb, transmission, atmospheric_light)
    
    # 다시 BGR로 변환하여 반환
    return cv2.cvtColor(dehazed_img_rgb, cv2.COLOR_RGB2BGR)

def find_furthest_road_point():
    """
    CCTV 이미지에서 안개를 제거하고, 도로를 분할하여 가장 먼 지점을 찾습니다.
    """
    # --- 1. 필요한 라이브러리 및 모델 준비 ---
    print("도로 세그멘테이션 모델(NVIDIA SegFormer)을 불러옵니다... (최초 실행 시 시간이 걸릴 수 있습니다)")
    try:
        # 호환성이 검증된 NVIDIA의 SegFormer 모델로 변경
        segmenter = pipeline("image-segmentation", model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        print("모델 로딩 완료.")
    except Exception as e:
        print(f"오류: 모델을 불러오는 중 문제가 발생했습니다. {e}")
        print("인터넷 연결을 확인하거나, 필요한 라이브러리가 모두 설치되었는지 확인해주세요.")
        return

    # --- 2. 이미지 불러오기 ---
    image_path = "image/fog_2.png"
    try:
        hazy_img = cv2.imread(image_path)
        if hazy_img is None:
            raise FileNotFoundError
        hazy_img_rgb = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        original_image_pil = Image.fromarray(hazy_img_rgb)
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: 이미지를 불러오는 중 문제가 발생했습니다. {e}")
        return

    # --- 3. 안개 제거 ---
    print("안개 제거를 시작합니다...")
    try:
        dehazed_img = dehaze_image_dcp(hazy_img)
        dehazed_img_pil = Image.fromarray(cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB))
        print("안개 제거 완료.")
    except Exception as e:
        print(f"오류: 안개 제거 중 문제가 발생했습니다. {e}")
        # 안개 제거 실패 시, 원본 이미지로 계속 진행
        dehazed_img_pil = original_image_pil
        print("안개 제거에 실패하여 원본 이미지로 분석을 계속합니다.")


    # --- 4. 도로 세그멘테이션 ---
    print("도로 영역 분할을 시작합니다...")
    try:
        # 디버깅을 위해 dehazed_img_pil 대신 original_image_pil 사용
        segments = segmenter(original_image_pil)
    except Exception as e:
        print(f"오류: 도로 세그멘테이션 중 문제가 발생했습니다. {e}")
        return
    
    road_mask = None
    print("--- 세그멘테이션 결과 라벨 목록 ---")
    for i, segment in enumerate(segments):
        label = segment['label']
        mask_img = segment['mask'].resize(original_image_pil.size)
        mask_np = np.array(mask_img)
        
        print(f"[{i}] 라벨: {label}, 마스크 크기: {mask_np.shape}, 마스크 고유 값: {np.unique(mask_np)}")
        
        # 모든 마스크를 디버깅용으로 저장
        debug_mask_path = f"image/debug_mask_{label.replace(' ', '_').replace('/', '_')}.png"
        Image.fromarray(mask_np).save(debug_mask_path)
        print(f"디버그 마스크를 '{debug_mask_path}'에 저장했습니다.")

        # 모델의 라벨이 'road, route'이므로 'road'가 포함되어 있는지 확인
        if 'road' in label:
            road_mask = mask_np
            print(f"'road' 라벨을 포함하는 마스크를 찾았습니다: {label}")
            
    if road_mask is None:
        print("분석 결과: 이미지에서 도로를 찾지 못했습니다.")
        return

    # --- 5. 최종 도로 세그멘테이션 마스크 저장 ---
    mask_image = Image.fromarray(road_mask)
    mask_output_path = "image/result_segmentation_mask.png"
    mask_image.save(mask_output_path)
    print(f"최종 도로 세그멘테이션 마스크를 '{mask_output_path}'에 저장했습니다.")

    # --- 6. 가장 먼 도로 지점 찾기 ---
    # 마스크에서 도로 픽셀(값이 0이 아닌 부분)의 좌표를 찾습니다.
    road_pixels_y, road_pixels_x = np.where(road_mask > 0)

    if len(road_pixels_y) == 0:
        print("분석 결과: 도로로 인식된 픽셀이 없습니다.")
        return

    # y 좌표가 가장 작은 픽셀이 가장 멀리 있는 지점입니다.
    min_y_index = np.argmin(road_pixels_y)
    furthest_point = (road_pixels_x[min_y_index], road_pixels_y[min_y_index])

    print(f"가장 먼 도로의 좌표를 찾았습니다: x={furthest_point[0]}, y={furthest_point[1]}")

    # --- 7. 결과 시각화 ---
    # 원본 이미지에 가장 먼 지점을 표시합니다.
    draw = ImageDraw.Draw(original_image_pil)
    # 지점을 잘 보이도록 원으로 표시합니다.
    radius = 10
    draw.ellipse(
        (furthest_point[0] - radius, furthest_point[1] - radius, 
         furthest_point[0] + radius, furthest_point[1] + radius),
        fill='red',
        outline='white',
        width=2
    )
    
    # 텍스트 추가
    try:
        font = ImageFont.truetype("malgun.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text((10, 10), f"Furthest Point: {furthest_point}", fill="red", font=font)


    # --- 7. 결과 출력 및 저장 ---
    output_path = "image/result_furthest_point.png"
    original_image_pil.save(output_path)
    print(f"결과 이미지를 '{output_path}'에 저장했습니다.")
    
    # 결과 이미지 보기
    original_image_pil.show()


if __name__ == '__main__':
    find_furthest_road_point()