# NTIRE 2026 官方退化管线(原样 vendored,**不要改**)

来源:CodaBench 竞赛页 *Data* 标签下的 "Transformations Script"
(`aug_utils_train.zip`,<https://drive.google.com/file/d/1oGr--PUOd11xy0ayYB6p2Mgg67n6eJPc>),
2026-08-26 取得,SHA1 见 `PROVENANCE.txt`。

四个文件(`__init__.py` / `distortions.py` / `utils_data.py` / `utils_distortions.py`)
与官方发布**逐字节相同**。这样做是为了让「我们施加的退化就是官方那一套」这句话可核验 ——
一旦改动,`utils/preprocess_ntire.py` 的分桶就不再等价于 NTIRE 的退化定义,
而这正是本轮实验的立论基础。要改行为请在 `preprocess_ntire.py` 里包一层,别动这里。

依赖:`torch` / `torchvision`(encode_jpeg/decode_jpeg)/ `kornia` / `scipy` / `numpy`。

官方内容摘要(`utils_data.py`):12 个畸变函数编成 7 组,每组 5 个强度档;
`get_distortions_composition` 无放回抽 1~3 个组,组内随机选变体,强度按高斯权重抽。
码本如何由此推出见 `utils/deg_taxonomy.py`。
