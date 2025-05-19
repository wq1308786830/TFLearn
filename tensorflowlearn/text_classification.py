import io
import os

import matplotlib.pyplot as plt
import sys
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

print(tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

""" start 此部分代码只需在首次运行时运行一次，后续运行时可注释掉 start """
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb_v1/aclImdb')

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())


remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

""" end 此部分代码只需在首次运行时运行一次，后续运行时可注释掉 end """


@tf.keras.utils.register_keras_serializable(package="Custom")
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb_v1/aclImdb/test',
    batch_size=batch_size)


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# for text in train_text.take(1):  # 打印前 5 条数据
#     print(text.numpy())
#
# for text, label in raw_train_ds.take(1):  # 打印前 5 条数据
#     print("Text:", text.numpy())
#     print("Label:", label.numpy())

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0)])

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# epochs = range(1, len(acc) + 1)
#
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
#
# plt.show()


export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(loss, accuracy)

export_model.save('text_classification_model.keras')


# loaded_model = tf.keras.models.load_model(
#     'text_classification_model.keras',
#     custom_objects={'Custom>custom_standardization': custom_standardization}
# )
#
# examples = [
#   "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge.",
#   """Years ago, when DARLING LILI played on TV, it was always the pan and scan version, which I hated and decided to wait and see the film in its proper widescreen format. So when I saw an inexpensive DVD of this Julie Andrews/Blake Edwards opus, I decided to purchase and watch it once and for all.<br /><br />Boy, what a terrible film. It's so bad and on so many levels that I really do not know where to start in describing where and when it goes so horribly wrong. Looking at it now, it's obvious to any fans of movies that Blake Edwards created this star vehicle for his wife simply because so many other directors had struck gold with Andrews in musicals (MARY POPPINS, SOUND OF MUSIC, THOROUGHLY MODERN MILLIE, etc) but also because Andrews was snubbed from starring in projects made famous on stage by Julie herself (CAMELOT, MY FAIR LADY, etc) because Hollywood thought she wasn't sexy or glamorous enough. So Blake created this stillborn effort, to showcase his wife in a bizarre concoction of spy story/war movie/romance/slapstick comedy/musical. DARLING LILI suffers from multiple personalities, never knowing who or what it is. Some specific scenes are good or effective but as a whole, it just doesn't work at all to a point of it being very embarrassing.<br /><br />Mind you, the version on the DVD is the "director's cut", or in this case, "let's salvage whatever we can" from this notorious box office flop. In releasing the DVD, Edwards cut 19 scenes (19!!!!!!!!) from the original bloated theatrical version into this more streamlined and yet remarkably ineffective version. The film moves along with no idea of what it is. We are 25 minutes into it and we still don't know what's going on or why we're watching what's going. What kind of spy is Lili? How powerful is she? Was she ever responsible for someone's death? Instead we watch a thoroughly bored looking Rock Hudson trying to woo a thoroughly bored looking Julie Andrews. Things aren't helped much with the inexplicable reason why the two fall in love. Why does Julie fall for Hudson? Why him and not other men she got involved with? There should have been one of her ex hanging around, trying to win her back or trying to decipher her secret. This would have given us some much needed contrast to the muddled action. It would also have given us some impetuous to the sluggish proceedings. There's no catalyst in this story.<br /><br />One only has to look at the cut scenes to clearly see that Edwards and the writer just came up with ideas inspired by Andrews' (and Edwards') previous successes. The best (or worst) example is the scene when Andrews and Hudson follows a group of children who sing in the middle of a forest. Edwards channeling SOUND OF MUSIC. It's no wonder he removed it from the DVD. Back in 1970, that scene might have worked on a certain level but today, that moment reeks of desperation. There are other plot elements directly inspired by Andrews/Edwards other films. The endless scenes of dogfights is inspired by the much better MODERN MILLIE. The musical moment "I'll give you three guesses" was created just to make fun of Julie's MARY POPPINS persona, which is turned "raunchy" with Julie doing a striptease in the act. The ending, bird's eye view of Julie running towards Hudson's plane, is another "wink" at SOUND OF MUSIC.<br /><br />The whole thing is confusing. Julie plays a singer, born from a German father and British mother, who lives in England but sings her (English) songs in Paris. You never know exactly where the story takes place. Some moments are just badly edited. Like when Julie and her "uncle" are on horseback. They talk and talk and then Julie suddenly sprints off in mid-sentence. I'm like "what happened here?"<br /><br />The comedy bits are unfunny and cringe-worthy. Every scene with the French police are pathetic. Where's Peter Sellers when you really need him. The action is stupid beyond belief. When Julie and her "uncle" are on their way to Germany on that train, Hudson's squadron shoots rounds of bullets at the train, almost killing Lili in the process. Brilliant. What's also funny about that scene is the two leave on the train in the middle of the night but Hudson and his squadron reach the train even though they fly off the next morning. That's one slow moving train there. <br /><br />The musical moments. The beginning is the best part of the entire film (and the reason I gave this film 3 stars) but it's effect is diminished considerably because it's repeated at the end. Speaking of redundant, did we really need to see a can-can dance, Crepe Suzette stripping scene and Julie stripping too? The "Girl in no man's land" is OK even if it's bleeding obvious, but that moment just doesn't make any sense whatsoever because Lili sings it to a group of injured soldiers at a French hospital, making me wonder: how many soldiers there were injured indirectly by the result of her spying?<br /><br />The whole project is listless and without energy. The romance is 100% unbelievable. Rock Hudson is way too old and tired looking (check out the museum scene). Julie looks dazed, like she's on Valium. But what really kills this ill-conceived project is Julie playing a German spy. Edwards desperately wanted to dispel the Mary Poppins syndrome afflicting his wife and believed that playing a traitor was a good career decision. As much as I like Julie, she's no Greta Garbo, who pulled it off so beautifully in MATA HARI. Funny enough, even if Julie plays a German spy, she still comes across as cloying and cute.<br /><br />How bad is DARLING LILI? Even after 37 years since its release, Blake Edwards felt he still needed to work on it for its DVD release.""",
#   """David Bryce's comments nearby are exceptionally well written and informative as almost say everything I feel about DARLING LILI. This massive musical is so peculiar and over blown, over produced and must have caused ruptures at Paramount in 1970. It cost 22 million dollars! That is simply irresponsible. DARLING LILI must have been greenlit from a board meeting that said "hey we got that Pink Panther guy and that Sound Of Music gal... lets get this too" and handed over a blank cheque. The result is a hybrid of GIGI, ZEPPELIN, HALF A SIXPENCE, some MGM 40s song and dance numbers of a style (daisies and boaters!) so hopelessly old fashioned as to be like musical porridge, and MATA HARI dramatics. The production is colossal, lush, breathtaking to view, but the rest: the ridiculous romance, Julie looking befuddled, Hudson already dead, the mistimed comedy, and the astoundingly boring songs deaden this spectacular film into being irritating. LILI is like a twee 1940s mega musical with some vulgar bits to spice it up. STAR! released the year before sadly crashed and now is being finally appreciated for the excellent film is genuinely is... and Andrews looks sublime, mature, especially in the last half hour......but LILI is POPPINS and DOLLY frilly and I believe really killed off the mega musical binge of the 60s..... and made Andrews look like Poppins again... which I believe was not Edwards intention. Paramount must have collectively fainted when they saw this: and with another $20 million festering in CATCH 22, and $12 million in ON A CLEAR DAY and $25 million in PAINT YOUR WAGON....they had a financial abyss of CLEOPATRA proportions with $77 million tied into 4 films with very uncertain futures. Maybe they should have asked seer Daisy Gamble from ON A CLEAR DAY ......LILI was very popular on immediate first release in Australia and ran in 70mm cinemas for months but it failed once out in the subs and the sticks and only ever surfaced after that on one night stands with ON A CLEAR DAY as a Sunday night double. Thank god Paramount had their simple $1million (yes, ONE MILLION DOLLAR) film LOVE STORY and that $4 million dollar gangster pic THE GODFATHER also ready to recover all the $77 million in just the next two years....for just $5m.... incredible!"""
# ]
# examples_tensor = tf.constant(examples)
#
# predictions = loaded_model.predict(examples_tensor)
# print(predictions)