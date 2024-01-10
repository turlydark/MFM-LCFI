# MFM-LCFI
This is the implementation of the article, Multiple Feature Mining Based on Local Correlation and Frequency Information for Face Forgery Detection.
## Abstract
As facial image manipulation techniques developed, deepfake detection attracted extensive attentions. Although re- searchers have made remarkable progresses in deepfake detection recently, which is still suffering from two limitations: a) current detectors achieve high accuracy in the high-quality videos and images, but it is hard to capture local and subtle artifacts in the low-quality and high-compression media; b) few of deeepfake detection methods gain satisfying performance under cross- database scenario, because detector overfit to specific color textures producing by same manipulation algorithm. Inspired the above issues, this paper proposes a novel framework fusing local related features and frequency information to mine the forgery patterns. Firstly, we design multi-feature enhancement module, which amplifies implicit local discrepancies and capture spatial correlation from three shallow feature layers and high- level semantic layer guided by attention maps. Secondly, dual frequency decomposition module is proposed for disassembling high-frequency and low-frequency features, the forgery artifacts are exposed after dual cross attention block processing in the frequency spectrum. Features from the two streams are fused to the classification for the final result. Comprehensive experiments demonstrate the superior performance of our proposed approach in the low-quality benchmark database and cross-dataset sce- nario.
## Index Terms
face forgery detection, deepfake manipulation, deepfake image detection, attention mechanism, multiple feature mining
