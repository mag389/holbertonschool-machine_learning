# the tokenizer can be also done with automodeling easier
from transformers import BertTokenizer, AutoModelForQuestionAnswering
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# that should cover the tokenizer portion
# model should be bert-uncased-tf2-qa from hub
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
# model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


# from: https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
  
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
