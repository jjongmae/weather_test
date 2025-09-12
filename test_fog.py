import cv2
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob

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
    
    num_pixels = int(h * w * top_pixels)
    indices = np.argsort(flat_dark_channel)[-num_pixels:]
    
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
    hazy_img_rgb = cv2.cvtColor(hazy_img_bgr, cv2.COLOR_BGR2RGB)
    dark_channel = get_dark_channel(hazy_img_rgb, patch_size=15)
    atmospheric_light = estimate_atmospheric_light(hazy_img_rgb, dark_channel)
    transmission = estimate_transmission(hazy_img_rgb, atmospheric_light, patch_size=15)
    dehazed_img_rgb = recover_scene_radiance(hazy_img_rgb, transmission, atmospheric_light)
    return cv2.cvtColor(dehazed_img_rgb, cv2.COLOR_RGB2BGR)

def process_image_and_find_point(image_path, output_dir, segmenter):
    """
    단일 CCTV 이미지에서 안개를 제거하고, 도로를 분할하여 가장 먼 지점을 찾습니다.
    """
    print(f"\n--- 처리 시작: {image_path} ---")
    
    base_filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(base_filename)[0]

    # --- 1. 이미지 불러오기 ---
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

    # --- 2. 안개 제거 ---
    print("안개 제거를 시작합니다...")
    try:
        dehazed_img = dehaze_image_dcp(hazy_img)
        dehazed_img_pil = Image.fromarray(cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB))
        print("안개 제거 완료.")
    except Exception as e:
        print(f"오류: 안개 제거 중 문제가 발생했습니다. {e}")
        dehazed_img_pil = original_image_pil
        print("안개 제거에 실패하여 원본 이미지로 분석을 계속합니다.")

    # --- 3. 도로 세그멘테이션 ---
    print("도로 영역 분할을 시작합니다...")
    try:
        segments = segmenter(original_image_pil)
    except Exception as e:
        print(f"오류: 도로 세그멘테이션 중 문제가 발생했습니다. {e}")
        return
    
    road_mask = None
    for segment in segments:
        label = segment['label']
        if 'road' in label:
            road_mask = np.array(segment['mask'].resize(original_image_pil.size))
            print(f"'road' 라벨을 포함하는 마스크를 찾았습니다: {label}")
            break # 도로를 찾으면 루프 종료
            
    if road_mask is None:
        print("분석 결과: 이미지에서 도로를 찾지 못했습니다.")
        return

    # --- 4. 최종 도로 세그멘테이션 마스크 저장 ---
    mask_image = Image.fromarray(road_mask)
    mask_output_path = os.path.join(output_dir, f"{filename_no_ext}_segmentation_mask.png")
    mask_image.save(mask_output_path)
    print(f"최종 도로 세그멘테이션 마스크를 '{mask_output_path}'에 저장했습니다.")

    # --- 5. 가장 먼 도로 지점 찾기 ---
    road_pixels_y, road_pixels_x = np.where(road_mask > 0)

    if len(road_pixels_y) == 0:
        print("분석 결과: 도로로 인식된 픽셀이 없습니다.")
        return

    min_y_index = np.argmin(road_pixels_y)
    furthest_point = (road_pixels_x[min_y_index], road_pixels_y[min_y_index])

    print(f"가장 먼 도로의 좌표를 찾았습니다: x={furthest_point[0]}, y={furthest_point[1]}")

    # --- 6. 결과 시각화 ---
    draw = ImageDraw.Draw(original_image_pil)
    radius = 10
    draw.ellipse(
        (furthest_point[0] - radius, furthest_point[1] - radius, 
         furthest_point[0] + radius, furthest_point[1] + radius),
        fill='red',
        outline='white',
        width=2
    )
    
    try:
        font = ImageFont.truetype("malgun.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text((10, 10), f"Furthest Point: {furthest_point}", fill="red", font=font)

    # --- 7. 결과 저장 ---
    output_path = os.path.join(output_dir, f"{filename_no_ext}_furthest_point.png")
    original_image_pil.save(output_path)
    print(f"결과 이미지를 '{output_path}'에 저장했습니다.")
    print(f"--- 처리 완료: {image_path} ---")


if __name__ == '__main__':
    INPUT_DIR = "image"
    OUTPUT_DIR = "output"

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

    # 처리할 이미지 파일 목록 가져오기 (png, jpg, jpeg)
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_paths:
        print(f"'{INPUT_DIR}' 폴더에 처리할 이미지가 없습니다.")
        exit()
        
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")

    # --- 모델 준비 ---
    print("\n도로 세그멘테이션 모델(NVIDIA SegFormer)을 불러옵니다... (최초 실행 시 시간이 걸릴 수 있습니다)")
    try:
        segmenter = pipeline("image-segmentation", model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        print("모델 로딩 완료.")
    except Exception as e:
        print(f"오류: 모델을 불러오는 중 문제가 발생했습니다. {e}")
        print("인터넷 연결을 확인하거나, 필요한 라이브러리가 모두 설치되었는지 확인해주세요.")
        exit()

    # --- 각 이미지 처리 ---
    for path in image_paths:
        process_image_and_find_point(path, OUTPUT_DIR, segmenter)
        
    print("\n모든 작업이 완료되었습니다.")
