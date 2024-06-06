#### Autoencoder techniques for survival analysis on renal cell carcinoma
A research evaluating the quality of the compression performed by different autoencoder models. On the one hand, the tabular autoencoder takes the tabular representation of transcriptomic data and can apply a series of penalties to add to the loss function. On the other hand, the graph autoencoder considers protein-protein interaction sources between genes to use as input graphs.

The penalties that these autoencoders can use are:

denoising: improves generalization of the data
sparse: finds meaningful latent representations
variational: fits a particular distribution to make the model generative
penalty combinations: we combined denoising and sparse to benefit from the advantages they both bring
Moreover, we added interpretability by finding the Mutual Information between the original transcriptomic data and the latent representations. The genes that were most expressed in the representations were LRP2 and ACE2, among a few others.
