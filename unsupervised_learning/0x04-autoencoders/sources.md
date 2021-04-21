[https://www.youtube.com/watch?v=FzS3tMl4Nsc&t=73s](autoencoder intro/definition)
Autoencoder: another type of neural network. It is a feed forward neural network.
Trained to reproduce it's input at the output layer. (as opposed to a class for instance)
Includes an encoder and decoder each a functin consisting of linear transform and 
a nonlinear transform (i.e. sigm(c + W * x)) 
tied weights mean weights for hidden layer and later layer are related W^ = WT (transpose)

[autoencoder loss function](https://www.youtube.com/watch?v=xTU79Zs4XKY)

need to compare X with X^ (then we can use gradient descent)

recall: for binary inputs we use cross-entropy
for real inputs we use l(f(x)) = .5 * sigma(X^ - X)^2 sum squared differences
	(with linear act function at ouput)

For autoencoders we have the gradient is just X^ - X
	use backpropagation for parameter gradients

[deep autoencoders](https://www.youtube.com/watch?v=z5ZYm_wJ37c)

undertraining is bg problem unless very large amount of data is used
but we get pretty good results with pre-training. (suprpising because usually
unsupervised pretraining is used to prevent over fitting not under)

Deep autoencoders can be used to reduce dimsionality of a data set.
for instance image compression. can be better than PCA (sinle layer network)

Another use is dataviz because we can disply hidden layres to display info

[deep autoencoders intro](https://www.jeremyjordan.me/autoencoders/)

Really great paper, too good and detailed to summarize here but it's not long and is plit nicely into sections.

[autoencoders in keras](https://blog.keras.io/building-autoencoders-in-keras.html)

[variational autoencoders](https://www.youtube.com/watch?v=fcvYpzHmhvA)

generative model that creates from input. Generative models produce from input.
Like the face creation website or creating picture of dogs.

normal autoencodres learn hidden representation of input. whereas VAE's generate new data

autoencoders minimize reconstruction loss. VAE minimize reconstruction loss + latent loss
latent loss is how the smapling is done to create the data.
last part of video is generative adversarial networks.

GAN's have generator and discriminator (similar to encoder/decoder) 

[another variational autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8)

the lss function is the same as normal autoencoers but also as latent erro which we use KL
for. Samples are taken with gaussian distribution.
we used partially fixed parts to enable backprop, that wat we can backprop with sampling.
