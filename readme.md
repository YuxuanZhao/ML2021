# Machine Learning

让机器找到一个复杂到很难直接编程得到的 function

<table>
  <tr>
    <td>Regression</td>
    <td>Scalar</td>
  </tr>
  <tr>
    <td>Classification</td>
    <td>Class</td>
  </tr>
  <tr>
    <td>Autoregressive Generation</td>
    <td>Infinite, structured data</td>
  </tr>
<table>

## Training Loss

<table>
  <tr>
    <td>训练损失很高的原因</td>
    <td>解决方案</td>
    <td>解释</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Model Bias</b>: model 本身的 limitation/ assumption 不足够表达/概括实际的情况</td>
    <td>Deeper, wider</td>
    <td>增加模型弹性</td>
  </tr>
  <tr>
    <td>Testing time scaling</td>
    <td>albert 的论文证明了重复同一个模型更长时间也会有更好的结果</td>
  </tr>
  <tr>
    <td rowspan="5"><b>Optimization</b>: 表达力够了，但优化时没找到目标。不同深度的网络对比训练损失，更深的网络有更多的训练损失证明没有训练到位</td>
    <td rowspan="2">Small Batch</td>
    <td>训练速度变慢，但是可以引入噪音，就不容易卡在 sharp minimal 和 saddle point，generalize 会使得 training 和 validate 同时变好</td>
  </tr>
  <tr>
      <td>Tayler Series Approximation 可以描述当前位置附近的 Loss function 形状：如果eigenvalue 全部为正就是 local minimal，经验上绝大多数训练卡住都是在 saddle point</td>
  </tr>
  <tr>
    <td rowspan="2">Adam: RMSProp + Momentum</td>
    <td>Momentum: 上一次移动方向 - 当前 gradient。虽然每次只考虑当前和上一次的 gradient，但是上一次的 gradient 也受到上上一次的影响：每一步都考虑了所有 gradient。会考虑方向和大小</td>
  </tr>
  <tr>
    <td>Adaptive Learning Rate: RMSProp。lr 太高根本没到不了 critical point。不同的参数 + 同一个参数不同的时候都需要有不同的 lr。只考虑大小，不考虑方向</td>
  </tr>
  <tr>
    <td>Warm up</td>
    <td>Bert, Transformer, RNN：一开始先慢慢走，到处看一下收集一点统计数据 sigma</td>
  </tr>
</table>

## Validation Loss

<table>
  <tr>
    <td>验证损失很高的原因</td>
    <td>解决方案</td>
    <td>解释</td>
  </tr>
  <tr>
    <td rowspan="3">overfitting: 在 training dataset 上有变好，但是在 unseen data 上变差了。模型能力太强，数据不够的地方是 freestyle</td>
    <td>更多的 training data (data augmentation)</td>
    <td>数据越多模型 freestyle 空间越少</td>
  </tr>
  <tr>
    <td>限制模型：less/ sharing parameters 减少 feature</td>
    <td>比如用 CNN 代替 FCN</td>
  </tr>
  <tr>
    <td>early stopping, regularization, dropout</td>
    <td>不改变网络架构的前提</td>
  </tr>
  <tr>
    <td>mismatch: 数据没收集好/data drift</td>
    <td>重新收集数据进行训练</td>
    <td>训练分布不同于测试分布</td>
  </tr>
</table>

## Classification

<table>
  <tr>
    <td>classification</td>
    <td>解释</td>
  </tr>
  <tr>
    <td>one-hot</td>
    <td>encode result</td>
  </tr>
  <tr>
    <td>softmax</td>
    <td>yi = exp(yi)/sum(exp(yi)) => normalize [0,1]</td>
  </tr>
  <tr>
    <td>sigmoid</td>
    <td>binary classification, 等同于 softmax</td>
  </tr>
  <tr>
    <td>cross entropy loss</td>
    <td>-sum(yi' ln yi), min CES = max Likelihood，比 MSE 更容易 training（error landscape 比较平坦）</td>
  </tr>
  <tr>
    <td>Feature Normalization</td>
    <td>相同的 gradient 变化在更大的输入下会对 loss 有更大的影响，所以输入应当做 standardization</td>
  </tr>
  <tr>
    <td>Batch Normalization</td>
    <td>做完之后，一个参数会对所有的参数有影响，所以我们需要一次性训练所有的数据。但这样不现实，所以我们使用足够大的 batch 来代表整个 corpus 的分布。能够改变 error surface 变得更平滑</td>
  </tr>
  <tr>
    <td>gamma, beta in BN</td>
    <td>一开始 gamma 为 1，beta 为 0。因为 bn 会给模型加上了 mean = 0, sd = 1 的限制，所以在一段时间的训练之后，模型可以慢慢放松这种限制（learnable gamma, beta）</td>
  </tr>
  <tr>
    <td>mu, sigma in eval</td>
    <td>直接使用训练时计算的 moving average of mu, sigma</td>
  </tr>
</table>

## CNN

<table>
  <tr>
    <td>CNN</td>
    <td>解释</td>
  </tr>
  <tr>
    <td>Receptive Field: kernel x channel</td>
    <td>只看一部分的图片信息</td>
  </tr>
  <tr>
    <td>Parameter sharing</td>
    <td>同一个任务的 filter 可以看所有的 receptive field</td>
  </tr>
  <tr>
    <td>Model Bias</td>
    <td>基于影像的，有非常 strong assumption，不能随意用于其他任务</td>
  </tr>
  <tr>
    <td>Feature Map</td>
    <td>把图片变成了一张新的图片，这张图片往往有更多的channel</td>
  </tr>
  <tr>
    <td>Pooling</td>
    <td>subsampling feature map 来节省资源</td>
  </tr>
  <tr>
    <td>Alpha Go</td>
    <td>棋盘表示成图片</td>
  </tr>
  <tr>
    <td>Data Augmentation</td>
    <td>CNN 不是 scale/ rotate invariant，spatial transformer layer 可以</td>
  </tr>
<table>

## Self Attention

<table>
  <tr>
    <td>self attention</td>
    <td>输入是可变长度的一系列向量</td>
  </tr>
  <tr>
    <td>Input</td>
    <td>One-hot embedding vs word embedding: text, voice, graph</td>
  </tr>
  <tr>
    <td>Output</td>
    <td>1:1 sequence labeling/ N:1 label sentiment analysis/ N: M seq2seq</td>
  </tr>
  <tr>
    <td>self attention query 如何考虑 key 的输入</td>
    <td>relevance (attention score): dot product（常用）, additive + tanh</td>
  </tr>
  <tr>
    <td>实际矩阵运算</td>
    <td>Q = W'I, K = W''I, V = W'''I, O = V(KQ)'</td>
  </tr>
  <tr>
    <td>Multi head self-attention</td>
    <td>相关这件事可能有很多种定义，所以QKV可以由不同的W来得到，那最后的维度可以通过再加一个FC转换成1</td>
  </tr>
  <tr>
    <td>Positional Encoding</td>
    <td>vanilla没有距离，可以有不同的方法生成一些和位置信息相关的数字加到原始输入上</td>
  </tr>
  <tr>
    <td>Truncated self attention</td>
    <td>如果sequence长度太长的话，模型参数会太多，所以我们可以人为设定一个能看得到的window</td>
  </tr>
  <tr>
    <td>Self Attention vs CNN</td>
    <td>CNN 只能看一个 receptive field，而 self attention 可以看整张图片，模型自己来决定哪些部分属于这个 pixel 的 receptive field</td>
  </tr>
  <tr>
    <td>Self Attention vs RNN</td>
    <td>RNN 只考虑左边已经输入的，有双向的 RNN 可以考虑全部，self attention 更能并行</td>
  </tr>
  <tr>
    <td>Self Attention in Graph</td>
    <td>没有 edge 的两个 node 可以设成 attention matrix = 0</td>
  </tr>
<table>

## Transformer

<table>
  <tr>
    <td>Transformer</td>
    <td>解释</td>
  </tr>
  <tr>
    <td>Seq2seq</td>
    <td>speech translation, machine translation, speech translation</td>
  </tr>
  <tr>
    <td rowspan="2">Encoder</td>
    <td>self attention: residual connection + layer norm</td>
  </tr>
  <tr>
    <td>FC: residual connection + layer norm</td>
  </tr>
    <tr>
    <td rowspan="2">Decoder</td>
    <td>Autoregressive: Begin, End token</td>
  </tr>
  <tr>
    <td>Non-autoregressive: 不是一个一个生成的，一次输入多个 Begin token。性能比 autoregressive 差，也更难训练好。那应该有多少个 Begin token 呢？<br>- 另一个 classifier来决定<br>- 给定超长的 Begin，停在 End 的时候</td>
  </tr>
  <tr>
    <td>Cross Attention</td>
    <td>decoder 的 self attention mask 产生出来的 query 和 encoder 的 k 和 v 做加权和。原始论文只使用最后一层 encoder 的结果，有很多别的研究做不同的 cross 方式</td>
  </tr>
  <tr>
    <td>Teacher forcing</td>
    <td>training 的时候使用 ground truth 作为 decoder 的输入</td>
  </tr>
  <tr>
    <td>Copy mechanism</td>
    <td>Summarization, point network</td>
  </tr>
  <tr>
    <td>Guided attention</td>
    <td>强迫 attention 必须有某种模式。monotonic attention, location-aware attention 应用：语音生成，语音辨识</td>
  </tr>
  <tr>
    <td>Beam Search</td>
    <td>Greedy decoding 不一定是最好的方法，beam search 也不一定更好，会影响创造力</td>
  </tr>
  <tr>
    <td>Optimize evalution metrics</td>
    <td>训练 cross entropy 不能直接推导到 BLEU，但是 BLEU 不能微分。不会 optimize 的时候，就用 RL 硬做</td>
  </tr>
  <tr>
    <td>Exposure bias</td>
    <td>Mismatch, scheduled sampling：会影响平行化的效果，引入实际生成的时候会看到的自己的比较差的输入</td>
  </tr>
</table>

## GAN

<table>
  <tr>
    <td>GAN Generator</td>
    <td>x + simple distribution = network => complex distribution</td>
  </tr>
  <tr>
    <td>Why distribution</td>
    <td>creativity: same input, different output</td>
  </tr>
  <tr>
    <td>unconditional generation</td>
    <td>只有 simple distribution (比如说 normal distribution)，没有 x</td>
  </tr>
  <tr>
    <td>Discriminator</td>
    <td>输入之前的 generator 的输出，输出有多像真的</td>
  </tr>
  <tr>
    <td>训练过程</td>
    <td>可以理解成模型来回进行两个过程，分别锁定generator层和discriminator层的参数，最小化loss和最大化loss</td>
  </tr>
  <tr>
    <td>怎么计算 distribution 的 divergence 呢？</td>
    <td>类似于 JS/KL Divergence 之类的，不好算，而且也不好微分。实际上直接使用 sampling，做二元分类的 cross entropy</td>
  </tr>
  <tr>
    <td>很难训练</td>
    <td>合理的图片是高维空间的 low-dim manifold，之间的交集可以忽略不计。如果sampling得不好，即使有相交，discriminator也很容易分辨</td>
  </tr>
  <tr>
    <td>没有重叠那有什么问题</td>
    <td>没有重叠的时候是没有区分度的，JS Divergence 永远都是 log2。直觉就是 classifier 几乎是 100%，只能人眼看效果好不好</td>
  </tr>
  <tr>
    <td>Wasserstein Distance</td>
    <td>想像成earth mover，穷举所有的把P的土全部移动到Q的计划的平均移动距离</td>
  </tr>
  <tr>
    <td>D in 1-Lipschitz</td>
    <td>D需要比较平滑，两个比较接近的点不能赋予差异很大的值。但是 Lipschitz 很难计算，所以有很多不同的方法去做类似的事情，比如说 clip 太大的 gradient，gradient penalty, 或spectral normalization</td>
  </tr>
  <tr>
    <td>GAN 做 sequence generation</td>
    <td>没办法微分，因为最后生成的是 token，小的delta可能不会影响选择哪个token。一个解决方法是用 RL</td>
  </tr>
  <tr>
    <td>GAN 没有公用的 metrics</td>
    <td>不同的task 设计不同的metrics，往往用另一个神经网络来判别(classifier)</td>
  </tr>
  <tr>
    <td>Mode Collapse</td>
    <td>generator 专门出 discriminator 的盲点的图片</td>
  </tr>
  <tr>
    <td>Mode Dropping</td>
    <td>generator 有多样性，但是其实只有训练数据的一部分分布，比如说只有黑人/白人/黄种人，这种 diversity 可以看Inception Score</td>
  </tr>
  <tr>
    <td>Frechet Inception Distance (Guassian)</td>
    <td>将真图片和生成的图片通过CNN在最后softmax前的结果当作是 gaussian distribution，衡量真和假的距离，越小越好，需要很多的sample</td>
  </tr>
  <tr>
    <td>Conditional Generator</td>
    <td>- 文生图：需要成对资料，negative sample 需要有乱配的，图片本来是真实的<br>- 图生图/声音生图</td>
  </tr>
  <tr>
    <td>Unsupervised learning</td>
    <td>Image Style Transfer，输入是 x domain 的图片的分布，输出是 y domain，discriminator 现在分辨是否是原本 y domain 的图片就行。但是 GAN 可能会无视这个 X 输入</td>
  </tr>
  <tr>
    <td>Cycle GAN</td>
    <td>训练两个 generator，一个把 X 转成 Y，一个把 Y 转成 X，解决 GAN 无视 X 输入的问题，实际上就算不用 cycle GAN，往往也是很像的</td>
  </tr>
</table>

## BERT

<table>
  <tr>
    <td>Self Supervised Learning (unsupervised)</td>
    <td>把输入的数据分成两部分，一部分用作 x，一部分用作 y<br>- Mask input：随机决定改成mask或者改成另一个随机字<br>- next sentence prediction：原版里有，后来被 RoBERTa 证明没什么用<br>- SOP Sentence Order Prediction: ALBERT 猜顺序稍微有点用</td>
  </tr>
  <tr>
    <td>Downstream Task</td>
    <td>一点点额外的label data就可以解很多下游任务（evaluation: GLUE 9个任务）<br>- Sequence => Class: Sentiment analysis<br>- Sequence => Sequence: POS tagging, extraction-based QA<br>- 2 Sequence => Class: Natural Language Inference (premise, hypothesis)</td>
  </tr>
  <tr>
    <td>BERT 的输出</td>
    <td>每个 token 在向量空间的 embedding，而且会同时考虑 context，一个词的意思和上下文极度相关(所以mask有用)<br>- fine-tuning on DNA/ protein/ music sequence classification</td>
  </tr>
  <tr>
    <td>Cross-lingual Alignment</td>
    <td>zero shot QA on different language: 不同的语言的相似token 可能在 BERT 的 embedding 也靠近在一起<br>- 但为什么？如果把语言的所有embedding求平均，然后计算两种语言的差值，我们可以加上这个差值直接做到 unsupervised token-level translation</td>
  </tr>
  <tr>
    <td>GPT</td>
    <td>预测下一个 token，使用 masked self attention。因为太大了，finetuning 也很困难，所以直接用 few shot learning (no gradient descent, in-context learning)，不用 finetuning 了</td>
  </tr>
  <tr>
    <td>CV</td>
    <td>Image: SimCLR, BYOL</td>
  </tr>
  <tr>
    <td>Speech</td>
    <td>GLUE - SUPERB</td>
  </tr>
</table>

## Auto Encoder

<table>
  <tr>
    <td>Auto Encoder</td>
    <td>类似于 Cycle GAN，把图片经过多层网络得到一个 vector (embedding/ representation/ code)，然后从这个 bottleneck 重建原图片</td>
  </tr>
  <tr>
    <td>为什么压缩图片是可行？</td>
    <td>如果我们随机设置像素，形成一张合理的图片的可能性很低，说明了其实合理图片是非常小的一个子集</td>
  </tr>
  <tr>
    <td>de-noise auto encoder</td>
    <td>图片，mask sentence</td>
  </tr>
  <tr>
    <td>feature disentangle</td>
    <td>输入一段音频，中间的 representation 能够分成不同的部分，比如说 content/ speaker information。那么我们就可以组合两个不同的声音的 content 和 speaker information 来做 voice conversion</td>
  </tr>
  <tr>
    <td>discrete representation</td>
    <td>原本的 vector 是 real number，可以改进成 binary 甚至是 one-hot<br>- VQVAE: code 转化成 code book 最近邻的 code 来输入 decoder<br>- seq2sea2seq: text representation，需要加一个 discriminator 来判定中间的 sequence 是否是人写的来避免学了暗号<br>- tree as embedding</td>
  </tr>
  <tr>
    <td>Variational Auto-Encoder (VAE)</td>
    <td>- encode to a Gaussian distribution<br>- sample from distribution, maximize Evidence Lower BOund (ELBO)<br>- Regularization: KL in loss generate a smooth, continuous manifold</td>
  </tr>
  <tr>
    <td>应用</td>
    <td>Anomaly Detection (Fraud, network intrusion, cancer detection)：没有什么异常例子的情况，单类分类。AE 对于 outlier 会生成出分布非常不正常的结果（无法 reconstruct）</td>
  </tr>
</table>

## Adversarial Attack

<table>
  <tr>
    <td>Adversarial Attack</td>
    <td>训练一个网络，对一个 benign image 增加一些人眼分辨不了的改变，targeted/ non-targeted 到另一个 class</td>
  </tr>
  <tr>
    <td>训练过程</td>
    <td>白盒下 gradient descent 到 image input 上而不是模型上</td>
  </tr>
  <tr>
    <td>loss</td>
    <td>L2 norm VS L-infinity：后者更好，因为人眼会对一个像素更大的改变更敏感。手动设置一个box范围，Fast Gradient Sign Method 直接用 box 的四个角，一拳超人，效果也不错</td>
  </tr>
  <tr>
    <td>Black box attack</td>
    <td>训练一个 network proxy（多个 ensemble 会更好），如果能骗到的话，可能也会骗到 black box</td>
  </tr>
  <tr>
    <td>特别的攻击手段</td>
    <td>one pixel, universal（万用的，而不是只有一个例子），也可以拓展到 speech, NLP 上，现实（人脸识别的眼镜干扰，交通标志），Adversarial reprogramming（劫持classifier做别的任务）, 混入精心设计的数据让模型有 back door</td>
  </tr>
  <tr>
    <td>防范</td>
    <td>smooth, compression, reconstruct, randomization, proactive defense（训练时产生 Adversarial 结果来学习）</td>
  </tr>
</table>

## ML Explainability

<table>
  <tr>
    <td>Explainable ML</td>
    <td>Loan issuer，不仅仅是答案，还要答案的理由<br>- interpretable: 本来模型就很简单，所以很容易解释和理解<br>- explainable: 本来模型很复杂，用常规手段无法理解，但是现在找到方法去理解<br>- Decision Tree 是不是兼具性能和可解释性呢？Random Forest 的复杂程度也不是人能够理解的<br>- 用简单模型distill 复杂模型，然后理解简单模型（Local Interpretable Model-Agnostic Explanations LIME）</td>
  </tr>
  <tr>
    <td>目标</td>
    <td>- 完整理解整个模型？<br>- Ellen Langer (The Copy Machine Study) 有理由就很有用，让人高兴就行</td>
  </tr>
  <tr>
    <td rowspan="2">Local Explanation (why this image is a cat?)</td>
    <td>输入<br>- Saliency Map：找到哪个点做轻微改动对 loss 的影响最大，数据集可能有一些奇怪的pattern被学到了（比如背景颜色和水标）<br>- SmoothGrad: 在图片上加很多次杂讯，然后求平均的Saliency Map<br>- Gradient Saturation: 微分=0就证明不重要吗？鼻子特别长的数据不会增加是大象的几率（因为已经是趋近于100%）改进是 Integrated Gradient</td>
  </tr>
  <tr>
    <td>网络<br>- Ablation study 消融实验：找到最重要的 component<br>- PCA/ t-SNE 降维<br>- 看 hidden layer 的结果，看 attention 的结果<br>- probing：训练模型去接收模型内部的 hidden layer 结果得出 class/ sequence</td>
  </tr>
  <tr>
    <td>Global Explanation (what does cat looks like?)</td>
    <td>Gradient Ascend, Generator，一般都不是人想象中的样子，要加很多限制，调很多超参数和新模型。其实我们不在乎机器在想什么。</td>
  </tr>
</table>

## Domain Adaptation (Transfer Learning)

<table>
  <tr>
    <td>Domain Drift</td>
    <td>Testing Data (Source Domain) 和 Training Data (Traget Domain) 的 distribution 不一样<br>- 输入从黑白变成彩色<br>- 输出从均匀变成倾斜<br>- 输入和输出的关系不一样</td>
  </tr>
  <tr>
    <td>Finetuning</td>
    <td>target domain data 很少，有 label</td>
  </tr>
  <tr>
    <td>Domain Adversarial Training</td>
    <td>target domain data 很多，没有 label<br>- 把模型分成两半，一半认为是 feature
     extractor，一半是 label predictor<br>- domain classifier: 类似于 GAN 判定图片是哪个 domain，可以把 feature extractor 想像成 generator，domain classifier 想像成 discriminator<br>- Decision Boundary (DIRT-T): unlabeled 的离 boundary 越远越好<br>- Universal Domain Adaptation: target domain 可能会有 source domain 里没有的类别</td>
  </tr>
  <tr>
    <td>Testing Time Training</td>
    <td>target domain data 很少，没有 label</td>
  </tr>
</table>

## Reinforcement Learning

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>