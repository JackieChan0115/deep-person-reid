本目录下的文件共分为两组：
1. floodfill.h floodfill.cpp FloodfillTool.cpp
2. sincurlimage.h sincurlimage.cpp SincurlImgTool.cpp

第一组的是实现漫水填充算法，以上文件经过编译后可以生成能够被python调用的共享动态库。
第二组的是实现图片曲线化调整姿态的算法，该组文件编译后，可以生成能够被python调用的共享动态库。

命令行分别为：
g++ -fPIC -shared -o FloodfillTool.so floodfill.cpp FloodfillTool.cpp
g++ -fPIC  -shared -o SincurlImgTool.so sincurlimage.cpp SincurImgTool.cpp

另外python调用的测试代码是imgtest.py
