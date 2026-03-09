# -*- coding: utf-8 -*-
"""
완전한 개선된 지폐+동전 이중 모델 통합 시스템
억지 동전 탐지 방지 + 신뢰도 0.5 이하 필터링 + UI 최적화
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class ImprovedDualMoneyDetector:
    def __init__(self, bill_model_path, coin_model_path, min_confidence=0.5):
        """
        개선된 이중 모델 화폐 탐지기
        
        Args:
            bill_model_path: 지폐+동전 통합 모델 경로
            coin_model_path: 동전 전용 모델 경로 (별도 학습)
            min_confidence: 최소 신뢰도 임계값 (기본값: 0.5)
        """
        # 최소 신뢰도 설정
        self.min_confidence = min_confidence
        print(f"최소 신뢰도 설정: {self.min_confidence}")
        
        # 화폐 금액 정의
        self.money_values = {
            '10won': 10, '50won': 50, '100won': 100, '500won': 500,
            '1000won': 1000, '5000won': 5000, '10000won': 10000, '50000won': 50000
        }
        
        # 색상 정의
        self.colors = {
            '10won': (255, 0, 0), '50won': (0, 255, 0), 
            '100won': (0, 0, 255), '500won': (255, 255, 0),
            '1000won': (255, 0, 255), '5000won': (0, 255, 255),
            '10000won': (128, 0, 128), '50000won': (255, 165, 0)
        }
        
        # 모델 로드
        try:
            self.bill_model = YOLO(str(bill_model_path))
            print(f"통합 모델 로드: {bill_model_path}")
            print(f"통합 모델 클래스: {self.bill_model.names}")
            
            self.coin_model = YOLO(str(coin_model_path))
            print(f"동전 전용 모델 로드: {coin_model_path}")
            print(f"동전 모델 클래스: {self.coin_model.names}")
            
            # 모델 타입 자동 감지
            self.is_combined_model = self._detect_model_type()
            print(f"주 모델 타입: {'통합 모델' if self.is_combined_model else '지폐 전용 모델'}")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def _detect_model_type(self):
        """모델이 통합모델인지 지폐전용인지 자동 감지"""
        bill_classes = self.bill_model.names
        
        # 동전 클래스가 포함되어 있으면 통합 모델
        coin_keywords = ['10won', '50won', '100won', '500won']
        has_coins = any(any(keyword in str(class_name).lower() for keyword in coin_keywords) 
                       for class_name in bill_classes.values())
        
        return has_coins
    
    def is_bill(self, class_name):
        """지폐인지 판단"""
        class_name_lower = str(class_name).lower()
        return any(bill in class_name_lower for bill in ['1000', '5000', '10000', '50000'])
    
    def is_coin(self, class_name):
        """동전인지 판단"""
        class_name_lower = str(class_name).lower()
        return any(coin in class_name_lower for coin in ['10won', '50won', '100won', '500won'])
    
    def extract_value(self, class_name):
        """클래스명에서 금액 추출"""
        class_name_str = str(class_name).lower()
        
        # 직접 매칭 시도
        if class_name_str in self.money_values:
            return self.money_values[class_name_str]
        
        # 패턴 매칭으로 금액 추출
        import re
        
        # 원화 패턴 (1000won, 5000won 등)
        won_match = re.search(r'(\d+)won', class_name_str)
        if won_match:
            value = int(won_match.group(1))
            if value in [10, 50, 100, 500, 1000, 5000, 10000, 50000]:
                return value
        
        # 숫자만 있는 경우
        number_match = re.search(r'(\d+)', class_name_str)
        if number_match:
            value = int(number_match.group(1))
            if value in [10, 50, 100, 500, 1000, 5000, 10000, 50000]:
                return value
        
        return None
    
    def detect_with_main_model(self, image):
        """통합 모델로 탐지 (신뢰도 필터링 적용)"""
        
        print(f"통합 모델로 탐지 중... (최소 신뢰도: {self.min_confidence})")
        
        # 통합 모델 설정 - 신뢰도를 낮게 설정하여 모든 후보를 받은 후 필터링
        results = self.bill_model(
            image,
            conf=0.1,        # 낮은 신뢰도로 모든 후보 받기
            iou=0.4,         
            max_det=50,      # 충분한 개수
            imgsz=1280,      # 높은 해상도
            augment=True,
            verbose=False
        )
        
        detections = []
        filtered_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"통합 모델 원시 탐지 결과: {len(boxes)}개")
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # 신뢰도 필터링 (0.5 이하 제거)
                    if conf < self.min_confidence:
                        filtered_count += 1
                        print(f"   신뢰도 부족으로 제외: {conf:.3f} < {self.min_confidence}")
                        continue
                    
                    # 실제 클래스명 가져오기
                    if cls in self.bill_model.names:
                        class_name = self.bill_model.names[cls]
                        print(f"   탐지: {class_name}")
                        
                        # 금액 추출
                        value = self.extract_value(class_name)
                        if value:
                            obj_type = 'bill' if self.is_bill(class_name) else 'coin'
                            
                            detections.append({
                                'class': str(class_name),
                                'confidence': conf,
                                'bbox': box,
                                'type': obj_type,
                                'value': value,
                                'color': self.colors.get(str(class_name), (255, 255, 255)),
                                'source': 'main_model'
                            })
                            print(f"   {obj_type} 인식: {class_name} = {value}원")
                        else:
                            print(f"   금액 추출 실패: {class_name}")
        
        if filtered_count > 0:
            print(f"신뢰도 필터링: {filtered_count}개 탐지 제외됨")
        
        return detections
    
    def detect_with_coin_model(self, image):
        """동전 전용 모델로 추가 탐지 (신뢰도 필터링 적용)"""
        
        print("동전 전용 모델로 추가 탐지 중...")
        
        # 1단계: 신뢰도 필터링을 적용한 확인
        quick_results = self.coin_model(
            image,
            conf=self.min_confidence,  # 설정된 최소 신뢰도 사용
            iou=0.4,         
            max_det=20,      
            imgsz=640,       
            verbose=False
        )
        
        # 확실한 동전이 있는지 먼저 확인
        confident_coins = 0
        for result in quick_results:
            if result.boxes is not None:
                confident_coins = len(result.boxes)
        
        print(f"확실한 동전 개수: {confident_coins}개 (신뢰도 >= {self.min_confidence})")
        
        # 확실한 동전이 없으면 정밀 탐지 생략
        if confident_coins == 0:
            print("확실한 동전이 없어서 정밀 탐지를 생략합니다.")
            return []
        
        # 2단계: 확실한 동전이 있을 때만 정밀 탐지
        print(f"{confident_coins}개의 확실한 동전 발견 -> 정밀 탐지 진행")
        
        # 동전을 위한 이미지 전처리
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        # 대비 강화 (동전 엣지 강조)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 정밀 탐지 (낮은 신뢰도로 받은 후 필터링)
        results = self.coin_model(
            enhanced,
            conf=0.1,        # 낮은 신뢰도로 모든 후보 받기
            iou=0.3,         
            max_det=50,      
            imgsz=1280,      
            augment=True,
            verbose=False
        )
        
        detections = []
        filtered_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"정밀 탐지 원시 결과: {len(boxes)}개")
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    # 업스케일링 보정 (좌표를 원본 크기로)
                    box = box // 2
                    
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # 신뢰도 필터링 (설정된 최소 신뢰도 적용)
                    if conf < self.min_confidence:
                        filtered_count += 1
                        print(f"   신뢰도 부족으로 제외: {conf:.3f} < {self.min_confidence}")
                        continue
                    
                    if cls in self.coin_model.names:
                        class_name = self.coin_model.names[cls]
                        print(f"   동전 탐지: {class_name}")
                        
                        # 동전만 처리
                        value = self.extract_value(class_name)
                        if value and self.is_coin(class_name):
                            detections.append({
                                'class': str(class_name),
                                'confidence': conf,
                                'bbox': box,
                                'type': 'coin',
                                'value': value,
                                'color': self.colors.get(str(class_name), (255, 255, 255)),
                                'source': 'coin_model'
                            })
                            print(f"   동전 인식: {class_name} = {value}원")
        
        if filtered_count > 0:
            print(f"동전 모델 신뢰도 필터링: {filtered_count}개 탐지 제외됨")
        
        return detections
    
    def detect_money(self, image_path_or_array, save_result=True, output_dir="improved_detection_results"):
        """개선된 통합 화폐 탐지 (신뢰도 0.5 이하 필터링)"""
        
        # 이미지 로드
        if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
            image = cv2.imread(str(image_path_or_array))
            image_name = Path(image_path_or_array).stem
        else:
            image = image_path_or_array.copy()
            image_name = "input_image"
        
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
        
        original_image = image.copy()
        
        print(f"개선된 이중 모델 분석 중: {image.shape}")
        print(f"신뢰도 필터링 적용: {self.min_confidence} 이상만 허용")
        
        # 1️⃣ 통합 모델로 탐지
        main_detections = self.detect_with_main_model(image)
        
        # 통합 모델에서 이미 동전을 충분히 찾았으면 동전 전용 모델 생략
        main_coins = [d for d in main_detections if d['type'] == 'coin']
        if len(main_coins) >= 3:  # 이미 3개 이상 동전을 찾았으면
            print(f"통합 모델에서 이미 {len(main_coins)}개 동전 발견 -> 동전 전용 모델 생략")
            coin_detections = []
        else:
            # 2️⃣ 동전 전용 모델로 추가 탐지 (놓친 동전 보완)
            coin_detections = self.detect_with_coin_model(image)
        
        # 3️⃣ 결과 통합 및 중복 제거
        all_detections = main_detections + coin_detections
        all_detections = self._remove_duplicates(all_detections)
        
        # 최종 검증: 의심스러운 동전 제거 + 신뢰도 재확인
        validated_detections = []
        for detection in all_detections:
            # 신뢰도 재확인
            if detection['confidence'] < self.min_confidence:
                print(f"최종 신뢰도 검증 실패: {detection['class']} ({detection['confidence']:.3f})")
                continue
            
            if detection['type'] == 'coin':
                # 동전의 크기 검증 (너무 크거나 작으면 제외)
                x1, y1, x2, y2 = detection['bbox']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # 동전 크기 범위 (픽셀 기준, 이미지 크기에 따라 조정 필요)
                min_area = 400   # 너무 작은 것 제외
                max_area = 30000  # 너무 큰 것 제외 (지폐는 보통 더 큼)
                
                if min_area <= area <= max_area:
                    validated_detections.append(detection)
                    print(f"동전 검증 통과: {detection['class']} (면적: {area})")
                else:
                    print(f"동전 크기 검증 실패: {detection['class']} (면적: {area})")
            else:
                validated_detections.append(detection)
                print(f"지폐 검증 통과: {detection['class']}")
        
        all_detections = validated_detections
        
        # 결과 분석
        coins = [d for d in all_detections if d['type'] == 'coin']
        bills = [d for d in all_detections if d['type'] == 'bill']
        total_amount = sum(d['value'] for d in all_detections)
        
        # 상세 결과 출력
        print(f"\n최종 검증된 탐지 결과 (신뢰도 >= {self.min_confidence}):")
        print(f"   지폐: {len(bills)}개")
        for bill in bills:
            print(f"      - {bill['class']} ({bill['source']})")
        
        print(f"   동전: {len(coins)}개")
        for coin in coins:
            print(f"      - {coin['class']} ({coin['source']})")
        
        print(f"   총 금액: {total_amount:,}원")
        
        # 시각화
        result_image = self._draw_detections(original_image, all_detections, total_amount)
        
        # 결과 저장
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            result_filename = f"{image_name}_confidence_{self.min_confidence}_{total_amount}won.jpg"
            cv2.imwrite(str(output_path / result_filename), result_image)
            print(f"결과 저장: {output_path / result_filename}")
        
        return {
            'detections': all_detections,
            'bills': bills,
            'coins': coins,
            'total_amount': total_amount,
            'bill_count': len(bills),
            'coin_count': len(coins),
            'result_image': result_image,
            'min_confidence': self.min_confidence
        }
    
    def _remove_duplicates(self, detections, overlap_threshold=0.3):
        """개선된 중복 탐지 제거"""
        
        if len(detections) <= 1:
            return detections
        
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # 교집합 계산
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # 합집합 계산
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # 신뢰도 순으로 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        for detection in detections:
            is_duplicate = False
            
            for filtered in filtered_detections:
                # 같은 클래스이고 겹치는 경우만 중복으로 판단
                if (detection['class'] == filtered['class'] and 
                    calculate_iou(detection['bbox'], filtered['bbox']) > overlap_threshold):
                    is_duplicate = True
                    print(f"중복 제거: {detection['class']}")
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _draw_detections(self, image, detections, total_amount):
        """탐지 결과 시각화 (이모지 제거, 신뢰도 숨김)"""
        
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            color = detection['color']
            value = detection['value']
            obj_type = detection['type']
            
            # 바운딩 박스 그리기 (타입별 두께 차별화)
            thickness = 4 if obj_type == 'bill' else 3
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 텍스트 (신뢰도 제거, 이모지 제거)
            label = f"{class_name}"
            value_text = f"{value:,}won"
            
            # 배경 그리기
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_image, (x1, y1 - text_height - 35), (x1 + text_width + 10, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(result_image, label, (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_image, value_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 총 금액 표시
        total_text = f"Total: {total_amount:,} WON"
        count_text = f"Bills: {len([d for d in detections if d['type'] == 'bill'])}, Coins: {len([d for d in detections if d['type'] == 'coin'])}"
        
        # 텍스트 크기 측정
        (total_text_width, total_text_height), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 배경 박스 크기 계산
        box_width = max(total_text_width, count_text_width) + 40
        box_height = total_text_height + count_text_height + 30
        
        cv2.rectangle(result_image, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
        cv2.rectangle(result_image, (10, 10), (10 + box_width, 10 + box_height), (0, 255, 0), 3)
        
        cv2.putText(result_image, total_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(result_image, count_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def _display_image_optimal_size(self, window_name, image, max_width=1920, max_height=1080):
        """이미지를 원본 크기에 맞게 표시하되, 1920x1080을 넘지 않도록 조정"""
        
        h, w = image.shape[:2]
        print(f"원본 이미지 크기: {w}x{h}")
        
        # 1920x1080을 넘는 경우에만 크기 조정
        if w > max_width or h > max_height:
            scale_w = max_width / w
            scale_h = max_height / h
            scale = min(scale_w, scale_h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"화면 크기에 맞게 조정: {w}x{h} -> {new_w}x{new_h} (스케일: {scale:.3f})")
        else:
            # 원본 크기 그대로 사용
            new_w, new_h = w, h
            resized_image = image
            print(f"원본 크기로 표시: {w}x{h}")
        
        # OpenCV 창 설정
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, new_w, new_h)
        cv2.moveWindow(window_name, 50, 50)  # 화면 좌상단에서 약간 떨어진 위치
        
        # 이미지 표시
        cv2.imshow(window_name, resized_image)
        
        return resized_image
    
    def set_confidence_threshold(self, new_threshold):
        """신뢰도 임계값 동적 변경"""
        old_threshold = self.min_confidence
        self.min_confidence = new_threshold
        print(f"신뢰도 임계값 변경: {old_threshold} -> {new_threshold}")

def main():
    """메인 실행 함수"""
    
    # 모델 경로 설정
    bill_model_path = "C:/Users/User/Downloads/Dataset/money_detection_coin_enhanced/weights/best.pt"  # 통합 모델
    coin_model_path = "C:/Users/User/Downloads/Dataset/money_detection_coin_enhanced/weights/best2.pt"  # 동전 전용 모델
    
    # 모델 경로 확인
    if not Path(bill_model_path).exists():
        print(f"통합 모델을 찾을 수 없습니다: {bill_model_path}")
        return
    
    if not Path(coin_model_path).exists():
        print(f"동전 모델을 찾을 수 없습니다: {coin_model_path}")
        print("동전 전용 모델을 별도로 학습해야 합니다!")
        return
    
    try:
        # 개선된 이중 모델 탐지기 초기화 (신뢰도 0.5로 설정)
        detector = ImprovedDualMoneyDetector(bill_model_path, coin_model_path, min_confidence=0.5)
        
        while True:
            print("\n개선된 이중 모델 화폐 탐지 시스템")
            print("=" * 50)
            print("1. 이미지 파일 분석")
            print("2. 실시간 카메라 탐지")
            print("3. 모델 정보 확인")
            print("4. 신뢰도 임계값 변경")
            print("5. 종료")
            
            choice = input("\n선택 (1-5): ").strip()
            
            if choice == '1':
                image_path = input("이미지 파일 경로: ").strip()
                if image_path and Path(image_path).exists():
                    try:
                        result = detector.detect_money(image_path)
                        
                        # 최적화된 크기로 표시
                        detector._display_image_optimal_size('Money Detection Result', result['result_image'])
                        
                        print("\n결과 이미지가 최적 크기로 표시되었습니다.")
                        print("아무 키나 누르면 창이 닫힙니다.")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                    except Exception as e:
                        print(f"분석 실패: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("파일이 존재하지 않습니다.")
            
            elif choice == '2':
                print("실시간 탐지 시작!")
                print("'q' 키를 눌러서 종료하세요.")
                
                # 실시간 창 설정
                window_name = 'Real-time Money Detection'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(window_name, 50, 50)
                
                cap = cv2.VideoCapture(1)
                
                if not cap.isOpened():
                    print("카메라를 열 수 없습니다.")
                    continue
                
                print("실시간 탐지 중...")
                
                while True:
                    ret, frame = cap.read()
                    if ret:
                        try:
                            # 성능을 위해 프레임 크기 조정
                            h, w = frame.shape[:2]
                            if w > 800:
                                scale = 800 / w
                                new_w, new_h = int(w * scale), int(h * scale)
                                frame = cv2.resize(frame, (new_w, new_h))
                            
                            result = detector.detect_money(frame, save_result=False)
                            
                            # 실시간 표시용 크기 조정
                            display_image = result['result_image']
                            h, w = display_image.shape[:2]
                            
                            # 실시간에서는 더 작은 크기로
                            if w > 1200 or h > 800:
                                scale = min(1200/w, 800/h)
                                new_w, new_h = int(w*scale), int(h*scale)
                                display_image = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                cv2.resizeWindow(window_name, new_w, new_h)
                            
                            cv2.imshow(window_name, display_image)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                                
                        except Exception as e:
                            # 오류 발생시 원본 프레임 표시
                            error_frame = frame.copy()
                            cv2.putText(error_frame, f"Error: {str(e)[:30]}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(error_frame, "Press 'q' to quit", (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            cv2.imshow(window_name, error_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    else:
                        print("프레임을 읽을 수 없습니다.")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                print("실시간 탐지 종료")
            
            elif choice == '3':
                print("\n모델 정보:")
                print(f"   통합 모델: {'통합 모델' if detector.is_combined_model else '지폐 전용 모델'}")
                print(f"   통합 모델 클래스: {detector.bill_model.names}")
                print(f"   동전 모델 클래스: {detector.coin_model.names}")
                print(f"   현재 신뢰도 임계값: {detector.min_confidence}")
            
            elif choice == '4':
                try:
                    current_threshold = detector.min_confidence
                    print(f"\n현재 신뢰도 임계값: {current_threshold}")
                    new_threshold = float(input("새로운 신뢰도 임계값 (0.0 ~ 1.0): ").strip())
                    
                    if 0.0 <= new_threshold <= 1.0:
                        detector.set_confidence_threshold(new_threshold)
                        print(f"신뢰도 임계값이 {new_threshold}로 변경되었습니다.")
                    else:
                        print("신뢰도는 0.0과 1.0 사이의 값이어야 합니다.")
                except ValueError:
                    print("올바른 숫자를 입력해주세요.")
            
            elif choice == '5':
                print("프로그램을 종료합니다.")
                break
            
            else:
                print("잘못된 선택입니다. 1-5 중에서 선택하세요.")
    
    except Exception as e:
        print(f"시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()