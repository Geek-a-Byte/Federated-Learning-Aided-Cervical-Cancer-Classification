# thesis
### ```status: Confused about what I am doing```
### ``TOC``
- [related papers](#related-papers)
- [github resources](#github-resources)
- [useful blogs](#blogs)
- [what is fl](#federated-learning)
- [useful videos](#youtube)
- [SO](#stackoverflow)

### ```github resources```

### ``Frameworks``

* EasyFL [[Github]](https://github.com/EasyFL-AI/EasyFL) [[Paper]](https://arxiv.org/abs/2105.07603)
* PySyft [[Github]](https://github.com/OpenMined/PySyft)
  * A Generic Framework for Privacy Preserving Peep Pearning [[Paper]](https://arxiv.org/abs/1811.04017)
* Tensorflow Federated [[Web]](https://www.tensorflow.org/federated)
* FATE [[Github]](https://github.com/FederatedAI/FATE)
* FedLearner [[Github]](https://github.com/bytedance/fedlearner) ByteDance
* Baidu PaddleFL [[Github]](https://github.com/PaddlePaddle/PaddleFL)
* Nvidia Clara SDK [[Web]](https://developer.nvidia.com/clara)
* [Flower.dev](https://flower.dev/)
* [OpenFL](https://github.com/intel/openfl)
* [FEDn](https://scaleoutsystems.github.io/fedn/) [[Github]](https://github.com/scaleoutsystems/fedn)
  * A modular and model agnostic framework for hierarchical federated machine learning [[Paper]](https://arxiv.org/abs/2103.00148)
  
### ``Healthcare``

* **The Future of Digital Health with Federated Learning** [[Paper]](https://www.nature.com/articles/s41746-020-00323-1)
  * General guide for FL in healthcare. Nice written paper.
* HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography [[Paper]](https://arxiv.org/abs/1909.05784) [NIPS 2019 Workshop]
* Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data [[Paper - Nature Scientific Reports 2020]](https://www.nature.com/articles/s41598-020-69250-1) [[News]](https://newsroom.intel.com/news/intel-works-university-pennsylvania-using-privacy-preserving-ai-identify-brain-tumors/)
* Learn Electronic Health Records by Fully Decentralized Federated Learning [[Paper]](https://arxiv.org/abs/1912.01792) [NIPS 2019 Workshop]
* Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records [[Paper]](https://arxiv.org/ftp/arxiv/papers/1903/1903.09296.pdf) [[News]](https://venturebeat.com/2019/03/25/federated-learning-technique-predicts-hospital-stay-and-patient-mortality/)
  * MIT CSAI, Harvard Medical School, Tsinghua University
* Federated learning of predictive models from federated Electronic Health Records. [[Paper]](https://www.ncbi.nlm.nih.gov/pubmed/29500022)
  * Boston University, Massachusetts General Hospital
* FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare [[Paper]](https://arxiv.org/pdf/1907.09173.pdf)
  * Microsoft Research Asia
* NVIDIA Clara Federated Learning to Deliver AI to Hospitals While Protecting Patient Data [[Blog]](https://blogs.nvidia.com/blog/2019/12/01/clara-federated-learning/)
  * Nvidia
* What is Federated Learning [[Blog]](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/)
  * Nvidia
* Split learning for health: Distributed deep learning without sharing raw patient data [[Paper]](https://arxiv.org/pdf/1812.00564)
* Two-stage Federated Phenotyping and Patient Representation Learning [[Paper]](https://www.aclweb.org/anthology/W19-5030.pdf) [ACL 2019]
* Federated Tensor Factorization for Computational Phenotyping [[Paper]](https://dl.acm.org/doi/10.1145/3097983.3098118) SIGKDD 2017
* FedHealth- A Federated Transfer Learning Framework for Wearable Healthcare [[Paper]](https://arxiv.org/abs/1907.09173) [ICJAI19 workshop]
* Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation [[Paper]](https://arxiv.org/abs/1810.04304) [MICCAI'18 Workshop] [Intel]
* Federated Patient Hashing [[Paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6121) [AAAI'20]
* Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data [[Paper]](https://arxiv.org/abs/1810.08553)
* Confederated Machine Learning on Horizontally and Vertically Separated Medical Data for Large-Scale Health System Intelligence [[Paper]](https://arxiv.org/abs/1910.02109)
* Privacy-Preserving Deep Learning Computation for Geo-Distributed Medical Big-Data Platform [[Paper]](http://www.cs.ucf.edu/~mohaisen/doc/dsn19b.pdf)
* Institutionally Distributed Deep Learning Networks [[Paper]](https://arxiv.org/abs/1709.05929)
* Federated semi-supervised learning for COVID region segmentation in chest CT using multi-national data from China, Italy, Japan [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841521000384)
- [Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/pdf/1810.04304.pdf)
- [Federated Learning in Distributed Medical Databases: Meta-Analysis of Large-Scale Subcortical Brain Data](https://arxiv.org/pdf/1810.08553.pdf)
- [Privacy-Preserving Technology to Help Millions of People: Federated Prediction Model for Stroke Prevention](https://arxiv.org/pdf/2006.10517.pdf)
- [A Federated Learning Framework for Healthcare IoT devices](https://arxiv.org/pdf/2005.05083.pdf)
Keywords: Split Learning + Sparsification
- [Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/pdf/2004.12321.pdf)
- [The Future of Digital Health with Federated Learning](https://arxiv.org/pdf/2003.08119.pdf)
- [Anonymizing Data for Privacy-Preserving Federated Learning. ECAI 2020.](https://arxiv.org/pdf/2002.09096.pdf)
- [Federated machine learning with Anonymous Random Hybridization (FeARH) on medical records](https://arxiv.org/pdf/2001.09751.pdf)
- [Stratified cross-validation for unbiased and privacy-preserving federated learning](https://arxiv.org/pdf/2001.08090.pdf)
- [Multi-site fMRI Analysis Using Privacy-preserving Federated Learning and Domain Adaptation: ABIDE Results](https://arxiv.org/pdf/2001.05647.pdf)
- [Learn Electronic Health Records by Fully Decentralized Federated Learning](https://arxiv.org/pdf/1912.01792.pdf)
- [Preserving Patient Privacy while Training a Predictive Model of In-hospital Mortality](https://arxiv.org/pdf/1912.00354.pdf)
- [Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270.pdf)
- [Federated and Differentially Private Learning for Electronic Health Records](https://arxiv.org/pdf/1911.05861.pdf)
- [A blockchain-orchestrated Federated Learning architecture for healthcare consortia](https://arxiv.org/pdf/1910.12603.pdf)
- [Federated Uncertainty-Aware Learning for Distributed Hospital EHR Data](https://arxiv.org/pdf/1910.12191.pdf)
- [Stochastic Channel-Based Federated Learning for Medical Data Privacy Preserving](https://arxiv.org/pdf/1910.11160.pdf)
- [Differential Privacy-enabled Federated Learning for Sensitive Health Data](https://arxiv.org/pdf/1910.02578.pdf)
- [LoAdaBoost: Loss-based AdaBoost federated machine learning with reduced computational complexity on IID and non-IID intensive care data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230706)
- [Privacy Preserving Stochastic Channel-Based Federated Learning with Neural Network Pruning](https://arxiv.org/pdf/1910.02115.pdf)
- [Confederated Machine Learning on Horizontally and Vertically Separated Medical Data for Large-Scale Health System Intelligence](https://arxiv.org/pdf/1910.02109.pdf)
- [Privacy-preserving Federated Brain Tumour Segmentation](https://arxiv.org/pdf/1910.00962.pdf)
- [HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/pdf/1909.05784.pdf)
- [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/pdf/1907.09173.pdf)
- [Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://arxiv.org/pdf/1903.09296.pdf)
- [LoAdaBoost:Loss-Based AdaBoost Federated Machine Learning on medical Data](https://arxiv.org/pdf/1811.12629.pdf)
- [FADL:Federated-Autonomous Deep Learning for Distributed Electronic Health Record](https://arxiv.org/pdf/1811.11400.pdf)

### ```related papers```
- [A Method of Federated Learning Based on Blockchain](https://dl.acm.org/doi/fullHtml/10.1145/3487075.3487143)
- [BVFLEMR: an integrated federated learning and blockchain technology for cloud-based medical records recommendation system](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-022-00294-6)
- [Blockchain for federated learning toward secure distributed machine learning systems: a systemic survey](https://link.springer.com/article/10.1007/s00500-021-06496-5)

### ``Youtube``

- [Terms: Independent and Identically Distributed (IID)](https://www.youtube.com/watch?v=EGKbPww2_rc)
- [Introduction to Neural Networks - The Nature of Code - video tuts](https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
- [fl videos](https://www.youtube.com/watch?v=2GCIjkqiorw&t=102s)

### ``Stackoverflow``

- [example for NON_IID Data](https://stackoverflow.com/questions/13058379/example-for-non-iid-data#:~:text=Literally%2C%20non%20iid%20should%20be,be%20decided%20by%20each%20other.)
- [what-is-the-use-of-verbose-in-keras-while-validating-the-model](https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model)

### ```blogs```

- [Work on the algorithm](https://www.tensorflow.org/federated/tutorials/building_your_own_federated_learning_algorithm)
- [Introduction to Neural Networks - The Nature of Code - web tuts](https://natureofcode.com/book/chapter-10-neural-networks/)
- [enumerate-in-python](https://www.geeksforgeeks.org/enumerate-in-python/)
- [top-resources-to-learn-about-federated-learning](https://analyticsindiamag.com/top-resources-to-learn-about-federated-learning/)
- [How to get started with FL](https://becominghuman.ai/federated-learning-collaborative-machine-learning-with-a-tutorial-on-how-to-get-started-2e7d286a204e)
- [first fl implementation to follow](https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399)
- [dl-fl-with-differential-privacy](https://xzhu0027.gitbook.io/blog/machine-learning/untitled/dl-fl-with-differential-privacy)


### ``Federated Learning``


Federated learning aims to train a single model from multiple data sources, under the constraint that data stays at the source and is not exchanged by the data sources (a.k.a. nodes, clients, or workers) nor by the central server orchestrating training, if present.

![image](https://user-images.githubusercontent.com/59027621/180593187-3ab0ef9f-39f8-40fd-a810-282c20418873.png)

In a typical federated learning scheme, a central server sends model parameters to a population of nodes (also known as clients or workers). The nodes train the initial model for some number of updates on local data and send the newly trained weights back to the central server, which averages the new model parameters (often with respect to the amount of training performed on each node). In this scenario the data at any one node is never directly seen by the central server or the other nodes, and additional techniques, such as secure aggregation, can further enhance privacy.

Imagine having access to text messages, emails, WhatsApp chats, Facebook and LinkedIn messages from millions of distinct accounts across the world in a bid to build a keypad next-word predictor. Or having an unrestricted access to billions of medical records across continents while predicting the chance of diabetes in a patient. These hypothetical scenarios underscore what data quantity and quality means in machine learning, but they are nowhere near being realisable in today’s world. Thanks to the tough data protection laws now in place nearly worldwide.

### Federated Learning (FL) to the rescue

![image](https://user-images.githubusercontent.com/59027621/180593638-9063af30-4204-4934-9196-2b1e3e4d102a.png)

Just as the image on the left, quality data exist like islands across edge devices around the world. But harnessing them into one piece to leverage their predictive power without contravening privacy laws is a herculean task. This challenge is what necessitated the FL technology. FL provides a clever way to connect machine learning models to the data required to effectively train them. So how does FL achieve this without breaching data protection laws? Read on as I take you through one of the hottest subjects in machine learning (ML) today.

The FL architecture in it’s basic form consists of a curator or server that sits at its centre and coordinates the training activities. Clients are mainly edge devices which could run into millions in number. These devices communicate at least twice with the server per training iteration. To start with, they each receive the current global model’s weights from the server, train it on each of their local data to generate updated parameters which are then uploaded back to the server for aggregation. This cycle of communication persists until a pre-set epoch number or an accuracy condition is reached. In the Federated Averaging Algorithm, aggregation simply means an averaging operation. That is all there is to the training of a FL model. I hope you caught the most salient point in the process — rather than moving raw data around, we now communicate model weights.

![image](https://user-images.githubusercontent.com/59027621/180596216-ce9c8290-46e9-4a8c-a708-a3d7b9b14287.png)
