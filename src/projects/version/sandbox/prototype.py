from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Labels placeholder—provide classes or generate pseudo-labels
y = df['label'].values

X = np.vstack(Q_esa)  # or combine Q_esa + A_esa

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=578)

inp = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(inp)
x = Dense(len(np.unique(y)), activation='softmax')(x)
model = Model(inp, x)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

