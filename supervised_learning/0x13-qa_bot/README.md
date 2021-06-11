# QA bot

Using transformers to accomplish real world ML tasks


For this project we need to install new packages:

pip install --user tensorflow==2.3

pip install --user tensorflow-hub

pip install --user transformers


Additional sources:

not directly applicable but useful: https://towardsdatascience.com/question-and-answering-with-bert-6ef89a78dac

nother way to make QA bot with bert: https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/

huggingface has a goood rundown of all the models, I used:

https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

and bert more generally:

https://huggingface.co/transformers/model_doc/bert.html

tf hub is where i drew most of it from in program:

https://tfhub.dev/see--/bert-uncased-tf2-qa/1

https://www.tensorflow.org/hub/tutorials/tf2_semantic_approximate_nearest_neighbors

https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
