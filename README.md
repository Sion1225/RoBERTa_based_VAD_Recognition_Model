#  [[IPSJ proceedings] A model based on RoBERTa for recognise VAD to sentences. (Siwon Seo, 2024)](https://www-ipsj-or-jp.translate.goog/event/taikai/86/WEB/data/pdf/7R-03.html?_x_tr_sl=ja&_x_tr_tl=ko&_x_tr_hl=ko&_x_tr_pto=sc)

Traditional emotion recognition was based on simple categorical classification, but to more accurately capture the complexity of human emotions, the use of the VAD model, which analyzes emotions in three dimensions: Valence (emotional value), Arousal (arousal level), and Dominance (dominance), has been proposed. In this study, we propose an efficient structure for a transformer model that recognizes VAD, using the EMOBank dataset and RoBERTa, a pre-trained large-scale language model.

### Guide
"Original_RoBERTa_V_A_D.py" is only one for the paper. 
Others are just traces of many trials and experiments.

### DataSet
Emo Bank [An Analysis of Annotated Corpora for Emotion Classification in Text](https://aclanthology.org/C18-1179) (Bostan & Klinger, COLING 2018)

Sven Buechel and Udo Hahn. 2017. EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. In EACL 2017 - Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics. Valencia, Spain, April 3-7, 2017. Volume 2, Short Papers, pages 578-585. Available: http://aclweb.org/anthology/E17-2092

Sven Buechel and Udo Hahn. 2017. Readers vs. writers vs. texts: Coping with different perspectives of text understanding in emotion annotation. In LAW 2017 - Proceedings of the 11th Linguistic Annotation Workshop @ EACL 2017. Valencia, Spain, April 3, 2017, pages 1-12. Available: https://sigann.github.io/LAW-XI-2017/papers/LAW01.pdf

### Reference
Paul Ekman. An argument for basic emotions. Cognition and Emotion, 6(3-4):169–200, 1992.

Margaret M. Bradley and Peter J. Lang. Affective norms for english words (anew): Instruction
manual and affective ratings. In Technical Report
C-1. The Center for Research in Psychophysiology, University of Florida., 1999.

Agnes Moors, Jan De Houwer, Dirk Hermans,
Sabine Wanmaker, Kevin van Schie, Anne-Laura
Harmelen, Maarten De Schryver, Jeffrey Winne,
and Marc Brysbaert. Norms of valence, arousal,
dominance, and age of acquisition for 4,300 dutch
words. Behavior research methods, 09 2012.

Sven Buechel and Udo Hahn. EmoBank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis. In Proceedings of the 15th Conference of the
European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers,
pages 578–585, Valencia, Spain, April 2017. Association for Computational Linguistics.

Sven Buechel and Udo Hahn. Readers vs. writers vs. texts: Coping with different perspectives
of text understanding in emotion annotation. In Proceedings of the 11th Linguistic Annotation
Workshop, pages 1–12, Valencia, Spain, April 2017. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee,
and Kristina Toutanova. Bert: Pre-training of
deep bidirectional transformers for language understanding, 2019.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du,
Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
Roberta: A robustly optimized bert pretraining
approach, 2019.

Diederik P. Kingma and Jimmy Ba. Adam: A
method for stochastic optimization, 2017.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023.

Sungjoon Park, Jiseon Kim, Seonghyeon Ye, Jaeyeol Jeon, Hee Young Park, and Alice Oh. Dimensional emotion detection from categorical emotion. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4367–4380, Online and Punta Cana,
Dominican Republic, November 2021. Association for Computational Linguistics.

Tom O’Malley, Elie Bursztein, James Long, Fran¸cois Chollet, Haifeng Jin, Luca Invernizzi, et al.
Kerastuner. https://github.com/keras-team/keras-tuner, 2019.

Mart´ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S.
Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew
Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath
Kudlur, Josh Levenberg, Dandelion Man´e, Rajat Monga, Sherry Moore, Derek Murray, Chris
Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vi´egas, Oriol Vinyals, Pete Warden, Martin
Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine
learning on heterogeneous systems, 2015. Software available from tensorflow.org.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick
von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger,
Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural
language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association
for Computational Linguistics
