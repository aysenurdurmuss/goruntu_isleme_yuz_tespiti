import time
import torch
import cv2
import threading
from ultralytics import YOLO


# Model hem "face" (0) hem "person" (1) sınıflarını doğrudan tanıyor
# NOT: Aynı ağırlık dosyasını İKİ AYRI nesne olarak yüklüyoruz.
# Sebebi: model.track() bir model nesnesine kalıcı olarak "takip mantığı" bulaştırıyor;
# aynı nesneyle sonra yapılan düz model() çağrısı da bundan etkileniyor.
# İki ayrı nesne kullanarak bu bulaşmayı önlüyoruz.
model = YOLO(r"C:\Users\aysen\Desktop\proje\runs\detect\train-8\weights\best.pt")        # SADECE bulanıklaştırma için (tracker'sız, düz tespit)
model_track = YOLO(r"C:\Users\aysen\Desktop\proje\runs\detect\train-8\weights\best.pt")  # SADECE ID ataması için (tracker'lı)


en_son_kare = None
en_son_blur_sonucu = None
en_son_track_sonucu = None

kare_kilidi = threading.Lock()
blur_kilidi = threading.Lock()
track_kilidi = threading.Lock()

durdurma_bayragi = threading.Event()

def blur_thread_fonksiyonu():
    global en_son_blur_sonucu #dıştaki en _son_blur_sonucu değişkenini kullanmak için 
    while not durdurma_bayragi.is_set(): #durdurma bayrağı kalkana kadar bu döngüye devam et 
        try: #sonraki satırlarda bir sorun olursa çökmesin except kısmına düşsün 
            with kare_kilidi: #kilidi bloğun sonuna otomatik bırak
                kare = en_son_kare #paylaşılan kareyi, kare adında geçici bir kopyaya al.
            if kare is None:  #henüz kamera hiç kare vermediyse model çağırmadan döngünü başına dön 
                continue
            sonuc = model(kare, conf=0.3, verbose=False) #model çağrısı

            with blur_kilidi:
                en_son_blur_sonucu = sonuc
            time.sleep(0.01)
        except Exception as hata:  #hata yakalama
            print("Blur thread hatası:", hata)


def track_thread_fonksiyonu():
    global en_son_track_sonucu 
    while not durdurma_bayragi.is_set():
        try:
            with kare_kilidi:
                kare = en_son_kare
            if kare is None:
                continue
            sonuc = model_track.track(kare, persist=True, verbose=False, conf=0.1, tracker="ozel_tracker.yaml")

            with track_kilidi:
                en_son_track_sonucu = sonuc
            time.sleep(0.01)
        except Exception as hata:
            print("Track thread hatası:", hata)
        
    


# Kameradan görüntü almak için VideoCapture nesnesi (0 = dahili kamera)
cap = cv2.VideoCapture(0)

ret, isinma_karesi = cap.read()
if ret:
    isinma_karesi = cv2.resize(isinma_karesi, (320, 240))
    model(isinma_karesi, conf=0.3, verbose=False)
    model_track.track(isinma_karesi, persist=True, verbose=False, conf=0.1, tracker="ozel_tracker.yaml")

blur_thread = threading.Thread(target=blur_thread_fonksiyonu, daemon=True)
track_thread = threading.Thread(target=track_thread_fonksiyonu, daemon=True)

blur_thread.start()
track_thread.start() 

# FPS hesaplamak için bir önceki karenin zamanını tutacak değişken
onceki_zaman = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Performans için görüntüyü küçült
    frame = cv2.resize(frame, (320, 240))

    with kare_kilidi:
        en_son_kare = frame.copy()

    with blur_kilidi:
        results_blur = en_son_blur_sonucu

    with track_kilidi:
        results_track = en_son_track_sonucu
        

    #  çizim,blur,ID yazma süresini ayrı ölçmek için
    cizim_baslangic = time.time()

    # bulanıklaştırma kutu çizme 
    if results_blur is not None:
        for result in results_blur:
            boxes = result.boxes.xyxy      # her kutunun koordinatları (x1, y1, x2, y2)
            classes = result.boxes.cls     # her kutunun sınıf numarası (0=face, 1=person)

            # boxes ve classes aynı sırada geldiği için zip
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)   # koordinatları tam sayıya çevir
                sinif_no = int(cls)              # sınıf numarasını tam sayıya çevir

                if sinif_no == 1:
                    # person sadece mavi kutu çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                elif sinif_no == 0:
                    # face bölgeyi kırp, bulanıklaştır, geri yaz
                    yuz_bolgesi = frame[y1:y2, x1:x2]

                    if yuz_bolgesi.shape[0] > 0 and yuz_bolgesi.shape[1] > 0:
                        bulanik_yuz = cv2.GaussianBlur(yuz_bolgesi, (99, 99), 0)
                        frame[y1:y2, x1:x2] = bulanik_yuz

    # id görünümü 
    if results_track is not None:
        for result in results_track:
            ids = result.boxes.id  #tracker ın verdiği id ler
            confs = result.boxes.conf  #her kutunun güven skoru
            if ids is not None:   # o karede onaylanmış hiçbir takip yoksa none gelir
                boxes = result.boxes.xyxy  #kutuların koordinatları
                classes = result.boxes.cls  #sınıf numaraları face=0 person=1
                for box, cls, obj_id, conf_score in zip(boxes, classes, ids, confs):
                    sinif_no = int(cls)
                    if sinif_no == 0:   # sadece face sınıfına id yaz
                        x1, y1, x2, y2 = map(int, box)
                        id_no = int(obj_id)
                        skor = float(conf_score)
                        cv2.putText(frame, f"ID:{id_no} ({skor:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #  çizim,blur,id yazma süresini burada kapanır ve hesaplanır
    cizim_bitis = time.time()
    cizim_suresi_ms = (cizim_bitis - cizim_baslangic) * 1000

    # FPS hesaplama ve yazdırma
    simdiki_zaman = time.time()
    fps = (
        1 / (simdiki_zaman - onceki_zaman)
        if (simdiki_zaman - onceki_zaman) > 0
        else 0
    )
    onceki_zaman = simdiki_zaman

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    #  çizim süresini de ekrana yaz 
    cv2.putText(
        frame,
        f"Cizim: {cizim_suresi_ms:.1f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Gokcelik Staj -  Optimizasyon", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

durdurma_bayragi.set()
blur_thread.join()
track_thread.join() 

cap.release()
cv2.destroyAllWindows()








