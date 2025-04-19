# Machine Learning

<table>
  <tr>
    <td>ML</td>
    <td>一个复杂到很难直接编程得到的 function</td>
  </tr>
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

## Training Loss 很高

<table>
  <tr>
    <td rowspan="2"><b>Model Bias</b>: model 本身的 limitation/ assumption 不足够表达/概括实际的情况</td>
    <td>Deep & Wide</td>
    <td>增加模型弹性</td>
  </tr>
  <tr>
    <td>Testing time scaling</td>
    <td>albert 的论文证明了同一个模型更长的inference时间也有更好的结果</td>
  </tr>
  <tr>
    <td rowspan="5"><b>Optimization</b>: 优化不了，可以对比不同深度的网络来确认</td>
    <td>Small Batch</td>
    <td>训练速度变慢，但可以引入噪音，就不容易卡在 sharp minimal 和 saddle point，generalize 会使得 training 和 validate 同时变好</td>
  </tr>
  <tr>
      <td>Tayler Series Approximation</td>
      <td>描述当前位置附近的 Loss function 形状，如果 eigenvalue 全部为正就是 local minimal，经验上绝大多数训练卡住都是在 saddle point</td>
  </tr>
  <tr>
    <td>Momentum</td>
    <td>上一次移动方向 - 当前 gradient。虽然每次只考虑当前和上一次的 gradient，但是上一次的 gradient 也受到上上一次的影响：每一步都考虑了所有 gradient。会考虑方向和大小</td>
  </tr>
  <tr>
    <td>RMSProp (Adaptive lr)</td>
    <td>lr 太高根本没到不了 critical point。不同的参数 + 同一个参数不同的时候都需要有不同的 lr。只考虑大小，不考虑方向</td>
  </tr>
  <tr>
    <td>Warm up</td>
    <td>一开始先慢慢走，到处看一下收集一点统计数据 sigma: Bert, Transformer, RNN</td>
  </tr>
</table>

## Validation Loss 很高

<table>
  <tr>
    <td rowspan="3">overfitting</td>
    <td>更多的 training data/ data augmentation</td>
    <td>数据越多模型 freestyle 空间越少</td>
  </tr>
  <tr>
    <td>限制模型：less/ sharing parameters 减少 feature</td>
    <td>比如用 CNN 代替 FCN</td>
  </tr>
  <tr>
    <td>early stopping, regularization, dropout</td>
    <td>不改变网络架构</td>
  </tr>
  <tr>
    <td>mismatch</td>
    <td>重新收集数据进行训练</td>
    <td>训练分布不同于测试分布，数据没收集好/data drift</td>
  </tr>
</table>

## Classification

<table>
  <tr>
    <td rowspan="2">Loss</td>
    <td>Sigmoid = Softmax</td>
    <td>binary: yi = exp(yi)/sum(exp(yi)) => normalize [0,1]</td>
  </tr>
  <tr>
    <td>Cross Entropy</td>
    <td>-sum(yi' ln yi), min CES = max Likelihood，landscape 比较平坦，比 MSE 更容易训练</td>
  </tr>
  <tr>
    <td rowspan="4">Batch Normalization</td>
    <td>原理</td>
    <td>相同的 gradient 变化在更大的输入下会对 loss 有更大的影响，所以输入应当做 standardization</td>
  </tr>
  <tr>
    <td>实现</td>
    <td>做 normalization 之后，每一个参数会对所有的参数有影响，所以我们需要一次性训练所有的数据。但这样不现实，所以我们使用足够大的 batch 来代表整个 corpus 的分布，这也能够改变 error surface 变得更平滑</td>
  </tr>
  <tr>
    <td>learnable gamma, beta</td>
    <td>一开始 gamma 为 1，beta 为 0。因为 bn 会给模型加上了 mean = 0, sd = 1 的限制，所以在一段时间的训练之后，模型可以慢慢放松这种限制</td>
  </tr>
  <tr>
    <td>Evaluation</td>
    <td>直接使用训练时计算的 moving average of mu, sigma</td>
  </tr>
</table>

## CNN

<table>
  <tr>
    <td rowspan="2">Input</td>
    <td>Receptive Field</td>
    <td>只看一部分的图片信息</td>
  </tr>
  <tr>
    <td>Alpha Go</td>
    <td>将棋盘表示成图片</td>
  </tr>
  <tr>
    <td>Output</td>
    <td>Feature Map</td>
    <td>把图片变成了一张有更多 channel 的新的图片</td>
  </tr>
  <tr>
    <td rowspan="2">Model Bias</td>
    <td>Parameter sharing</td>
    <td>同一个任务的 filter 可以看所有的 receptive field</td>
  </tr>
  <tr>
    <td>Pooling</td>
    <td>subsampling feature map 来节省资源</td>
  </tr>
  <tr>
    <td>Data Augmentation</td>
    <td>scale/ rotate invariant</td>
    <td>CNN 不是，spatial transformer layer 是</td>
  </tr>
<table>

## Self Attention

<table>
  <tr>
    <td>Input</td>
    <td>可变长度的很多 embedding</td>
    <td>text, voice, graph</td>
  </tr>
  <tr>
    <td rowspan="3">Output</td>
    <td>1:1</td>
    <td>sequence labeling</td>
  </tr>
  <tr>
    <td>N:1</td>
    <td>label sentiment analysis</td>
  </tr>
  <tr>
    <td>N: M</td>
    <td>seq2seq</td>
  </tr>
  <tr>
    <td rowspan="3">query 如何考虑 key</td>
    <td>Relevance (attention score)</td>
    <td>常用 dot product，也可以 additive + tanh</td>
  </tr>
  <tr>
    <td>矩阵运算</td>
    <td>Q = W'I, K = W''I, V = W'''I, O = V(KQ)'</td>
  </tr>
  <tr>
    <td>in Graph</td>
    <td>unconnected node 的 attention matrix 设为 0</td>
  </tr>
  <tr>
    <td rowspan="3">Trick</td>
    <td>Multi head</td>
    <td>相关可以有很多定义，所以 QKV 可以由不同的 W 来得到，最后通过一个FC转换成一维</td>
  </tr>
  <tr>
    <td>Positional Encoding</td>
    <td>vanilla 的没有距离，可以有不同的方法生成一些和位置信息相关的数字加到原始输入上</td>
  </tr>
  <tr>
    <td>Truncated</td>
    <td>如果sequence长度太长的话，模型参数会太多，可以人为设定一个能看得到的 window</td>
  </tr>
  <tr>
    <td rowspan="2">优点</td>
    <td>CNN</td>
    <td>CNN 只能看一个 receptive field，而 self attention 可以看整张图片，模型自己来决定哪些部分属于这个 pixel 的 receptive field</td>
  </tr>
  <tr>
    <td>RNN</td>
    <td>RNN 只考虑左边已经输入的，有双向的 RNN 可以考虑全部，self attention 更能并行</td>
  </tr>
<table>

## Transformer

<table>
  <tr>
    <td>Seq2seq</td>
    <td>应用</td>
    <td>speech translation, machine translation, speech translation</td>
  </tr>
  <tr>
    <td rowspan="2">Encoder</td>
    <td>self attention</td>
    <td>residual connection + layer norm</td>
  </tr>
  <tr>
    <td>FC</td>
    <td>residual connection + layer norm</td>
  </tr>
    <tr>
    <td rowspan="4">Decoder</td>
    <td>Autoregressive</td>
    <td>Begin, End token</td>
  </tr>
  <tr>
    <td rowspan="2">Non-autoregressive</td>
    <td>不是一个一个生成的，一次输入多个 Begin token。性能比 autoregressive 差，也更难训练</td>
  </tr>
  <tr>
    <td>Begin token 数量：<br>- 由另一个 classifier来决定<br>- 给定超长的 Begin，停在 End 的时候</td>
  </tr>
  <tr>
    <td>Teacher forcing</td>
    <td>training 的时候使用 ground truth 作为 decoder 的输入</td>
  </tr>
  <tr>
    <td>Cross Attention</td>
    <td>Decoder Query + Encoder Key, Value</td>
    <td>做加权和，原始论文只使用最后一层 encoder 的结果，有很多别的研究做不同的 cross 方式</td>
  </tr>
  <tr>
    <td rowspan="2">Guided attention</td>
    <td>模式</td>
    <td>monotonic attention, location-aware attention</td>
  </tr>
  <tr>
    <td>应用</td>
    <td>语音生成，语音辨识</td>
  </tr>
  <tr>
    <td rowspan="4">Trick</td>
    <td>Beam Search</td>
    <td>Greedy decoding 不一定最好，beam search 也不一定更好，会影响创造力</td>
  </tr>
  <tr>
    <td>Goal unmatched</td>
    <td>训练 cross entropy 不能直接推导到 BLEU，但 BLEU 又不能微分，可以用 RL</td>
  </tr>
  <tr>
    <td>Copy mechanism</td>
    <td>Summarization, point network</td>
  </tr>
  <tr>
    <td>Mismatch, scheduled sampling</td>
    <td>引入实际生成的时候会看到的自己的比较差的输入，可以减少 Exposure bias，但会影响平行化的效果</td>
  </tr>
</table>

## GAN

<table>
  <tr>
    <td>Generator</td>
    <td>same input, different output</td>
    <td>x + simple distribution = network => complex distribution, distribution 可以带来 creativity</td>
  </tr>
  <tr>
    <td>Discriminator</td>
    <td>Classifier</td>
    <td>输入之前的 generator 的输出，输出有多像真的</td>
  </tr>
  <tr>
    <td>Unconditional</td>
    <td>没有 x</td>
    <td>只有 simple distribution (比如说 normal distribution)</td>
  </tr>
  <tr>
    <td rowspan="4">Conditional</td>
    <td>文生图</td>
    <td>需要成对资料，negative sample 需要有乱配的，图片本来是真实的</td>
  </tr>
  <tr>
    <td>声音生图</td>
    <td>水声越大，瀑布越猛</td>
  </tr>
  <tr>
    <td>Image Style Transfer</td>
    <td>输入是 x domain 的图片的分布，输出是 y domain，discriminator 现在分辨是否是原本 y domain 的图片就行。但是 GAN 可能会无视这个 X 输入</td>
  </tr>
  <tr>
    <td>Cycle GAN</td>
    <td>训练两个 generator，一个把 X 转成 Y，一个把 Y 转成 X，解决 GAN 无视 X 输入的问题，实际上就算不用 cycle GAN，往往也是很像的</td>
  </tr>
  <tr>
    <td rowspan="7">Training</td>
    <td>过程</td>
    <td>来回进行：锁定generator层和discriminator层的参数，最小化loss和最大化loss</td>
  </tr>
  <tr>
    <td>怎么计算 distribution 的 divergence</td>
    <td>类似于 JS/KL Divergence，不好算，而且也不好微分。实际上直接使用 sampling，做二元分类的 cross entropy</td>
  </tr>
  <tr>
    <td>为什么难以训练</td>
    <td>合理的图片是高维空间的 low-dim manifold，之间的交集可以忽略不计。如果sampling得不好，即使有相交，discriminator也很容易分辨。没有重叠的时候是没有区分度的，JS Divergence 永远都是 log2。直觉就是 classifier 几乎是 100%，只能人眼看效果好不好</td>
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
    <td>Sequence Generation</td>
    <td>没办法微分，因为最后生成的是 token，小的 delta 不能影响到选中哪个 token，用 RL</td>
  </tr>
  <tr>
    <td>Frechet Inception Distance (Guassian)</td>
    <td>将真图片和生成的图片通过CNN在最后softmax前的结果当作是 gaussian distribution，衡量真和假的距离，越小越好，需要很多的sample</td>
  </tr>
  <tr>
    <td rowspan="3">模型问题</td>
    <td>No public metrics</td>
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
</table>

## BERT

<table>
  <tr>
    <td rowspan="3">Self Supervised Learning (unsupervised)</td>
    <td>Mask input</td>
    <td>随机决定改成mask或者改成另一个随机字</td>
  </tr>
  <tr>
    <td>next sentence prediction</td>
    <td>原版里有，后来被 RoBERTa 证明没什么用</td>
  </tr>
  <tr>
    <td>SOP Sentence Order Prediction</td>
    <td>ALBERT 猜顺序稍微有点用</td>
  </tr>
  <tr>
    <td rowspan="4">Downstream Task</td>
    <td>Evaluation</td>
    <td>GLUE 9个任务</td>
  </tr>
  <tr>
    <td>Sequence => Class</td>
    <td>Sentiment analysis</td>
  </tr>
  <tr>
    <td>Sequence => Sequence</td>
    <td>POS tagging, extraction-based QA</td>
  </tr>
  <tr>
    <td>2 Sequence => Class</td>
    <td>Natural Language Inference (premise, hypothesis)</td>
  </tr>
  <tr>
    <td rowspan="2">Output</td>
    <td rowspan="2">Embedding in Context</td>
    <td>一个词的意思和上下文极度相关(所以mask有用)</td>
  </tr>
  <tr>
    <td>fine-tuning on DNA/ protein/ music sequence classification</td>
  </tr>
  <tr>
    <td rowspan="2">Cross-lingual Alignment</td>
    <td>zero shot QA on different language</td>
    <td>不同的语言的相似token 可能在 BERT 的 embedding 也靠近在一起</td>
  </tr>
  <tr>
    <td>为什么？</td>
    <td>如果把语言的所有embedding求平均，然后计算两种语言的差值，我们可以加上这个差值直接做到 unsupervised token-level translation</td>
  </tr>
  <tr>
    <td rowspan="4">应用</td>
    <td rowspan="2">NLP: GPT</td>
    <td>预测下一个 token，使用 masked self attention</td>
  </tr>
  <tr>
      <td>因为太大了，finetuning 也很困难，所以直接用 few shot learning (no gradient descent, in-context learning)</td>
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
    <td>有些任务很难label(不知道哪个是最优解)</td>
    <td>RL 三步走：Observation, Action, Reward<br>- Policy Network (Actor): 输入游戏画面，输出 action 的概率，然后 sample<br>- Define Loss: -Total Reward = loss<br>- Optimization: Reward 和 Environment 都是黑盒，actor 和 environment 都有随机性。类比 GAN 的话，我们不知道 discriminator 的结构，也不能做 gradient descent</td>
  </tr>
  <tr>
    <td>Policy gradient (Control actor)</td>
    <td>{e, a} 都对应一个我们有多希望在这种情况下机器做/不做某件事，那怎么评价呢？<br>1. 评价一个 action，我们需要考虑之后发生的所有 reward (cumulated): Reward Delay<br>2. 很远的 reward 也有关系吗？discounted accumulation gamma^N<br>3. reward 是相对的，所以需要normalization -b<br>4. 用 critic 的结果来当作 b，也就是说比预想中的平均值好才行<br>5. Advantage Actor-Critic: A = r + V+1 - V 也就说说用作出了a得到的t+1这个时刻的和所有t这个时刻的V来比</td>
  </tr>
  <tr>
    <td>On-policy VS Off-policy</td>
    <td>背景：一般数据只更新一个 epoch，每次更新都需要收集新的资料（同一个操作，结果对于不同能力的模型来说可能不一样的，能力低的模型驾驭不了高难度的操作）<br>- on-policy: interact 和 train 的 actor 最好是同一个<br>- off-policy 能省很多收集资料的时间 (Proximal Policy Optimization PPO), 直觉是 train 的模型要知道它跟 interact 的模型是不同的（而且可以量化这种不同）<br>- Exploration: 如果 actor 从来没做过某种行为，那么我们永远也不知道那件事好还是不好，所以 actor in interact 应该增大随机性</td>
  </tr>
  <tr>
    <td>Critic</td>
    <td>Value function: s + 观察对象模型 theta => 预测 discounted cumulated reward<br>- Monte-Carlo: 看完一整个 episode，认为先后两个s是有关系的，TD则认为无关<br>- Temporal-Difference (TD): 只需要看 s, a, r, s+1 就足够了，适合很长的游戏，训练的时候因为知道 r，所以可以用前后两个 s 作为输入的差值<br>- Actor 和 critic 可以 shared bottom<br>- Deep Q Network: 只使用 critic</td>
  </tr>
  <tr>
    <td>Sparse Reward</td>
    <td>绝大多数时候都没有 reward，添加 extra reward to guide (reward shaping)，需要 domain knowledge<br>- Curiosity: meaningful new</td>
  </tr>
  <tr>
    <td>No Reward</td>
    <td>Imitation Learning: expert demonstration/ behavior cloning 机器可能会学不到失败的情况，可能不知道哪个行为是essential，哪个是 irrelevant<br>- Inverse Reinforcement learning: 用 expert demonstration 来学 reward function，然后正常学 actor，假设老师的行为是最好的，在生成 trajectories 的时候，reward function必须给老师的行为更高的reward。这个过程中的 actor 其实是 generator，reward function 是 discriminator</td>
  </tr>
</table>

## Life Long Learning (Incremental)

<table>
  <tr>
    <td>Catastrophic interference (forgetting)</td>
    <td>把线上资料和过往所有资料混在一起更新？计算和存储都是问题，multitask training 一般会用这个混合的数据集的训练结果作为 upper bound</td>
  </tr>
  <tr>
    <td>每个任务都有独立模型不行吗？</td>
    <td>不现实，不能融会贯通，人脑不就可以吗？对比 transfer learning 关心新的任务能不能从旧的任务上汲取信息, life long learning 更关心旧的任务还能做得一样好</td>
  </tr>
  <tr>
    <td>Evaluation</td>
    <td>画一个accuracy matrix，然后可以看有没有 transfer learning无师自通，也可以看有没有 catastrophic interference学了新的忘了旧的<br>最终 measure 可以用平均 accuracy 或者 backward transfer（差值）或者 forward transfer</td>
  </tr>
  <tr>
    <td>解决方案</td>
    <td>- Selective Synaptic Plasticity：<br>
    1. EWC: Loss 增加每个参数都有一个 guard b（人为设置的），如果这个参数在之前的任务非常重要，就不许大幅改动。类似的有: SI, MAS, RWalk, SCP<br>2. Gradient Episodic Memory (GEM): gradient 更新的方向会考虑之前的任务的 gradient，需要之前的资料来计算 gradient<br>- Additional Neural Resource Allocation: <br>1. Progressive Neural Network (额外任务就增加额外模型)<br>2. packNet（先做大模型，然后每次都只使用一部分参数）<br>3. CPG: 前面两者的结合<br>- Memory reply: 每个数据/任务都训练一个数据 generator，非常有效<br>- curriculum learning：任务学习的先后顺序</td>
  </tr>
</table>

## Network Compression

<table>
  <tr>
    <td>原因：edge device/ privacy</td>
    <td>大的模型比较好 train（大乐透假说，initialize参数的正负号很重要，大小不重要，大模型可能完全不用训练里面就已经有 subnetwork 是正确的）</td>
  </tr>
  <tr>
    <td>Network Prunning</td>
    <td>反复：根据 weight（现实中很难实现一个 neuron 不给另外一个特定 neuron 传参数，而且硬件加速不了反而会变慢）/ neuron （直接改模型的 dimension）的重要性剪掉，然后 fine tune</td>
  </tr>
  <tr>
    <td>Distillation</td>
    <td>小模型不学 ground truth，而是模仿大模型的结果（可能是一个 distribution），这个大模型甚至可以是 ensemble<br>- Temperature T: 平滑化 softmax，让学生更容易学（不然就等同于 ground truth）</td>
  </tr>
  <tr>
    <td>Parameter Quantization</td>
    <td>- FP16, FP8, binary weight<br>- weight clustering,可以在训练的时候就要求 weight 比较接近<br>- Hufffman-encoding</td>
  </tr>
  <tr>
    <td>Depthwise Separable Convolution</td>
    <td>- 几个 channel 就有几个 kernel（普通的 CNN 是可以不同的）每个 kernel 只负责自己对应的那一个 channel（不会交叉）<br>- 1x1 conv 来做 point-wise channel 间的关系<br>为什么有用？Low-rank approximation 的概念：网络变深反而参数量变少，会减少 rank</td>
  </tr>
  <tr>
    <td>Dynamic Computation</td>
    <td>- dynamic depth: 每一个 layer 都可以加一个 extra layer 直接得出结果，MSDNet 改进了这个<br>- dynamic width: Slimmable Neural Networks<br>- 模型自行决定：SkipNet,BlockDrop 根据输入的难度决定</td>
  </tr>
</table>

## Meta learning (learn to learn)

<table>
  <tr>
    <td>训练一个模型来找最好的 hyper parameter (network architecture, parameters, learning rate)</td>
    <td>1. 使用很多不同的 task 的 dataset (support set = training, query set = validation) 来训练一个 meta learning network<br>2. 真正想要的不同的任务作为 testing task，使用这些 task 的很小的 train (within-task training)/ validation (within-task testing) dataset 就能找到最优的参数（loss 可以用 cross entropy）3. learning to compare (matric-based) 可以只有一个步骤，直接得到结果而不是模型</td>
  </tr>
  <tr>
    <td>学 initialize model: Model-Agnostic Meta-Learning (MAML) 优点是可以 feature reuse, Reptile</td>
    <td>Pretrained Model(self-supervised learning), domain adaptation/ transfer learning 也是类似的观点 ，可以把所有的 task 数据混在一起来训练作为 baseline</td>
  </tr>
  <tr>
    <td>Network Architecture Search (NAS)</td>
    <td>- RL: discrete number 类似于 # filter, height, width, stride, ...<br>- DARTS: 把不能微分的模型变成可以</td>
  </tr>
  <tr>
    <td>Data Augmentation</td>
    <td>sample reweighting</td>
  </tr>
  <tr>
    <td>应用</td>
    <td>Few-shot image classification: n-way k-shot (Omniglot dataset)</td>
  </tr>
</table>