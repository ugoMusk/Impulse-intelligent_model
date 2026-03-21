import os
import tensorflow as tf
from model.transformer import IIMoModel
from training.data_loader import DataLoader
from utils.config import Config


def create_loss_fn(pad_token_id):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    )

    def loss_fn(y_true, y_pred):
        loss = loss_object(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss_fn


def train():
    config = Config("configs/model_config.yaml")

    data_loader = DataLoader(
        data_dir="data/processed",
        tokenizer_path="backend/model/tokenizer.model",
        max_seq_len=config.max_seq_length
    )

    train_dataset = data_loader.build_dataset("train", config.batch_size)
    val_dataset = data_loader.build_dataset("validation", config.batch_size, shuffle=False)

    model = IIMoModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_length
    )

    pad_token_id = 0  # ensure tokenizer pad_id == 0 or pass dynamically
    loss_fn = create_loss_fn(pad_token_id)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )

    model.compile(optimizer=optimizer, loss=loss_fn)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/iimo_{epoch}",
            save_weights_only=True,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/iimo_model")

    print("✅ Training complete.")


if __name__ == "__main__":
    train()