# Created by: Luke Richards
# Purpose: This is the implementation of the GLS system, this shows the desired end result. This code as of now will not run as it is pseudo-code written by Frank Ferraro

from GroundedLanguageLearning import *

# Nisha AAAI 2018

# vocab_size = ; ## determine how many words to learn
# rgb_feat_size = 3
#
# aaai18_rgb_model = GroundedLanguageClassifier(
#   text_encoder=ExmptyExtractor(),
#   percept_encoder=RGBFeatureExtractor(rgb_feat_size), ## not defined here
#   corr_scorer=MultiLabelBinaryMLPScorer(rgb_feat_size, vocab_size)
# )
# gls_nisha = GLSLearner(
#   model=aaai18_rgb_model,
#   loss=nn.BCELoss()
# )

# Luke RSS 2019

vocab_size = 500 ## determine how many words to learn
mutlimodal_feat_size = 51

rss19_cnn_model = GroundedLanguageClassifier(
  text_encoder=EmptyExtractor(),
  percept_encoder=CNNFeatureExtractor("data/UW_lukeRSS2019_cnn_object_features.csv", mutlimodal_feat_size),
  corr_scorer=MultiLabelBinaryMLPScorer(mutlimodal_feat_size, vocab_size)
  # this is for the alignment part
)
# gls_luke = GLSLearner(
#   model=rss19_cnn_model,
#   loss=nn.BCELoss()
# )


# gls.train(train_data, val_data, ...)
