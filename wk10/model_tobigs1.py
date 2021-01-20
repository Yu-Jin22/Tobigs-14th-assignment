import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_datasets as tfds

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = tf.keras.Sequential()
    
        # (100,)를 input으로 받아서 7*7*256짜리로 만들고 이를 reshape해준다. 
        self.model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Reshape((7, 7, 256)))
        assert self.model.output_shape == (None, 7, 7, 256) #output_shape이 (None, 7, 7, 256)이게아니면 에러를 띄우라는 의미이다!

        self.model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) #noise추가로 이미지를 생성하기 위해 upsampling층이용
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.LeakyReLU())

        self.model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)
        #마지막층의 활성화함수만 tanh이고 나머지 층은 LeakyReLU()를 활성화함수로 사용한다. 
        
    def call(self, inputs, training =True) :
        return self.model(inputs)



class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = tf.keras.Sequential()
    
        #[28, 28, 1]크기의 이미지를 받아 Conv연산을 진행한다
        self.model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.3))

        self.model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU())
        self.model.add(tf.keras.layers.Dropout(0.3))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1)) #이미지가 진짜인지 가짜인지를 판단하기 위해서 결과는 스칼라값으로 나온다. 
    
    def call(self, inputs, training =True) :
        return self.model(inputs)
   
#========================================================================================#

def discriminator_loss(loss_object, real_output, fake_output):
    #here = tf.ones_like(????) or tf.zeros_like(????)  -> tf.zeros_like와 tf.ones_like에서 선택하고 (???)채워주세요
    real_loss = loss_object(tf.ones_like(real_output), real_output) #real이미지를 넣으면 1이 나오고 fake이미지를 넣으면 0이 나올수 있도록한다
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)

def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image


def save_imgs(epoch, generator, noise):
    gen_imgs = generator(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig.savefig("images/mnist_%d.png" % epoch)

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='/data/tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # settting hyperparameter
    latent_dim = 100
    epochs = 2
    batch_size = 10000
    buffer_size = 6000
    save_interval = 1

    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(lr = 0.0002,beta_1 = 0.5, beta_2 = 0.9) #학습의 안정화를 위해 lr과 beta_1을 내렸다
    disc_optimizer = tf.keras.optimizers.Adam(lr = 0.0002,beta_1 = 0.5, beta_2 = 0.9)

    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) #이진분류사용하여 cross_entropy를 계산한다

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = generator(noise) #noise를 추가하여 fake이미지를 만든다

            real_output = discriminator(images)#real과 fake를 넣어 결과값을 도출한다
            generated_output = discriminator(generated_images)

            gen_loss = generator_loss(cross_entropy, generated_output) #cross_entropy를 사용하여 loss값을 뽑는다
            disc_loss = discriminator_loss(cross_entropy,real_output, generated_output)

        grad_gen = tape.gradient(gen_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss

    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0

        for images in train_dataset:
            gen_loss, disc_loss = train_step(images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print('Time for epoch {} is {} sec - gen_loss = {}, disc_loss = {}'.format(epoch + 1, time.time() - start, total_gen_loss / batch_size, total_disc_loss / batch_size))
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, seed)


if __name__ == "__main__":
    train()