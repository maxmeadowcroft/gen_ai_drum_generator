from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow import keras

@register_keras_serializable(package="Custom")
class SamplingLayer(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable(package="Custom")
class VAE(keras.Model):
    def __init__(self, encoder, decoder, original_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.original_dim = original_dim

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
        reconstruction_loss *= self.original_dim
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return reconstructed

    def get_config(self):
        config = super().get_config()
        config["encoder"] = keras.utils.serialize_keras_object(self.encoder)
        config["decoder"] = keras.utils.serialize_keras_object(self.decoder)
        config["original_dim"] = self.original_dim
        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        decoder_config = config.pop("decoder")
        original_dim = config.pop("original_dim")

        encoder = keras.utils.deserialize_keras_object(encoder_config)
        decoder = keras.utils.deserialize_keras_object(decoder_config)

        model = cls(encoder=encoder, decoder=decoder, original_dim=original_dim, **config)
        return model

# Example Encoder
encoder_inputs = keras.Input(shape=(384,))
x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
x = keras.layers.Dense(128, activation='relu')(x)
z_mean = keras.layers.Dense(32)(x)
z_log_var = keras.layers.Dense(32)(x)
z = SamplingLayer()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Example Decoder
decoder_inputs = keras.Input(shape=(32,))
x = keras.layers.Dense(128, activation='relu')(decoder_inputs)
x = keras.layers.Dense(256, activation='relu')(x)
decoder_outputs = keras.layers.Dense(384, activation='sigmoid')(x)
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

# Instantiate VAE
vae = VAE(encoder, decoder, original_dim=384)
vae.compile(optimizer='adam', loss=None)  # Loss is added via add_loss in call()

vae.summary()
vae

vae.save('vae_model.keras')
