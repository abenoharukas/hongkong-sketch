# 香港岛 / Hong Kong Island

一张可以拖动缩放的手绘香港地图，做法照
[How Isometric Pixel City Was Made](https://www.yadongxie.com/blog/how-pixel-singapore-was-made)：
一台正交相机把 Google 的三维瓦片拍成一整幅底图，再整幅交给图像模型重绘，
最后切成瓦片在浏览器里看。

    hk/source/plate/      Google 三维瓦片的原始正交截图，24 个窗口 1024x1024
    hk/source/hk-*.jpg    过程中留下的样张
    hk/plate.png          拼好、刷过水色的整幅底图（6144x4096，48 MB，不进 git）
    hk/flat-source.jpg    喂给图像模型的平涂色块底图（2400x1600）
    hk/redraw.png         模型的输出（1872x1248）
    hk/corner-redraw.png  同角重画一张，用来盖掉模型的水印
    hk/art.png            成图（1858x1234）
    hk/tiles/             Deep Zoom 金字塔 z0..z2，给 hk/map.html 用
    hk/ortho.html         一台正交相机，Cesium 里画 Google 的三维瓦片
    hk/shoot.mjs          无头 Chrome 逐个窗口截图（CDP）
    hk/serve.mjs          同源静态服务器（Google 的瓦片要真的 Origin）
    hk/build.py           拼版、体检、水色、色块化、修水印、切瓦片
    hk/diffuse.py         MultiDiffusion 采样（本地 SDXL + 风格 LoRA）
    hk/map.html           查看器

详细说明在 [hk/README.md](hk/README.md)。

## 看图

    node hk/serve.mjs 8130
    open http://127.0.0.1:8130/hk/map.html

## 重拍底图

    node hk/shoot.mjs --out hk/source/plate --cols 6 --rows 4 --w 1024 --h 1024 \
      --mpp 1.1 --lon 114.1600 --lat 22.2770 --az -15 --tilt 45
    .venv/bin/python hk/build.py --check

需要 `.env` 里的 `GOOGLE_MAPS_API_KEY`（`hk/key.js` 从它生成，已被 git 忽略）。

## 换图

任何一张 1872x1248 左右的新画面丢进去就能出全套瓦片：

    .venv/bin/python hk/build.py --redraw 新图.png

底图和成图都是三维影像重绘的衍生物，仅供技术演示，不能当地图用。
三维影像 © Google Maps。
