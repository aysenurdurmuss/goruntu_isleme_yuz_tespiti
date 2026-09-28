# Yüz Bulanıklaştırma ve Nesne Takip Sistemi

Canlı kamera görüntüsü üzerinde insan yüzlerini gerçek zamanlı olarak tespit edip bulanıklaştıran ve tespit edilen kişileri takip eden bir bilgisayarlı görü uygulaması. Bir staj projesi kapsamında geliştirilmiştir.

## Özellikler

- Özel eğitilmiş bir YOLOv8 modeli ile "face" ve "person" sınıflarının tespiti
- ByteTrack ile tespit edilen kişilerin takip edilerek kimlik (ID) ataması
- Tespit edilen yüzlerin Gaussian Blur ile bulanıklaştırılması
- Gerçek zamanlı performans için threading (çoklu iş parçacığı) mimarisi
- Anlık FPS (kare hızı) ölçümü ve ekranda gösterimi

## Kullanılan Teknolojiler

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

## Veri Seti Hakkında

Model eğitiminde kullanılan ham görseller bu depoda yer almamaktadır (görseller internetten toplanmıştır). Veri seti AnyLabeling ile etiketlenip `labelme2yolo` aracıyla YOLO formatına dönüştürülmüştür.
