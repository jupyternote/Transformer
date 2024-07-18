手动复现的完整的transformer代码，用于学习目的


原创性声明：
  1.代码为完全原创，并非来自别人的项目
  2.main函数中有chatgpt实现的部分，但是与模型实现无关
  
测试声明：
  1.没有找到合适的语言数据集，因此进行图像生成任务，在mian函数中有对任务测试的详细描述
  2.使用mnist数据集，并针对图像生成进行了针对模型的少量修正
  3.测试使用了恒源云显卡（租用），3090，运行时间约30分钟
  4.测试代码并不完善，实际测试并非使用上传的main函数，而是略有更改

代码结构：（代码实现中有较为详细的注释以及单元测试）
  1.embedding进行初始编码，如位置编码
  2.self_attention实现了基本的注意力层
  3.encoder/decoder调用self_attention，实现了编/解码器基本结构
  4.Transformer包含完整的transformer结构
  5.main函数进行调用和测试
  6.process文件夹不用管它，删了也行。本来是处理文本、想进行文本训练的，但是无奈文本数据集质量不佳，还有一堆脏话，无奈放弃

致歉
  1.本人第一次提交github项目，刚刚学会使用，因此没有中间提交过程
  2.代码在云端进行了少量调整，因此项目可能存在些许运行错误。但是很容易debug
      

以下是测试得到的结果之一（不知道图片能不能加载出来）
![train_epoch4_batch3000_img1](https://github.com/user-attachments/assets/863b059e-372d-46cc-93cd-62a958d4a4f5)

![train_epoch4_batch3000_img2](https://github.com/user-attachments/assets/c05f1d36-2ebf-468f-a094-2ef9a488b0f8)

![train_epoch4_batch3000_img5](https://github.com/user-attachments/assets/280fd0ae-5057-4ef8-8a29-496359157689)

![train_epoch4_batch3000_img4](https://github.com/user-attachments/assets/f4830859-4e77-4905-bb67-da6a12bad67f)
