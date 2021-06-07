# OCR服务器文档

## 功能
此服务器是客户端的接入服务器，功能为识别客户上传照片含有的字段，并像客户端输出
* 每个字段含有的字体
* 每个字段在照片里的位置
* 对输出每个字段中识别的的字体的置信度

## 输入定义（由客户端发送）
客户端需要用 POST, HTTP/1.1 或 HTTP/1.0, multipart/form-data 向【服务器地址】发送请求。
请求中需要向POST请求的files上传一个张key = "img"的图片。

## 输出定义（由客户端接取）
接收请求随后，服务器会往客户端发送一件UTF-8 JSON文件。此文件的格式为：
{
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"numBlocks": n
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"texts": [text1，text2， ..., textn]
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "blocks": [[[x11, y11], [x21, y21], [x31, y31], [x41, y41]], [[x12, y12], [x22, y22], [x32，y32], [x42, y42]], ... 
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[x1n, y1n], [x2n, y2n], [x3n, y3n], [x4n, y4n]]]
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"scores": [score1, score2, ... scoren]
    }
其中的参数为下:
|  参数名   |  类型  |注释 |
| :-------: | :----: | -------- | 
| numBlocks | integer |从图片里面识别出的字段数量 |
| texts | 数组 |含有n个识别出来的字段 (String) |
| scores |数组 | 含有n个double。其中, scorem 代表着 textm的置信度，数值为0-1|
| blocks | 数组 | 含有n个子数组。其中，第m个子数组代表着textm在图片中的定位|

blocks 中每个子数组的格式和定义如下：
[[x1m, y1m], [x2m, y2m], [x3m, y3m], [x4m, y4m]]
|  参数名   |  类型  |注释 |
| :-------: | :----: | -------- | 
| x1m | integer | 字段m的左上角离照片左边的x-距离 |
| y1m | integer | 字段m的左上角离照片上边的y-距离 |
| x2m | integer | 字段m的右上角离照片左边的x-距离 |
| y2m | integer | 字段m的右上角离照片上边的y-距离 |
| x3m | integer | 字段m的右下角离照片左边的x-距离 |
| y3m | integer | 字段m的右下角离照片上边的y-距离 |
| x4m | integer | 字段m的左下角离照片左边的x-距离 |
| y4m | integer | 字段m的左下角离照片上边的y-距离 |