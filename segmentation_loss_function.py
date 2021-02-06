import tensorflow as tf
import keras.backend as K


# --------------------------------- weighted cross entropy -------------------------------
def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        return tf.reduce_mean(loss)

    return loss


# --------------------------------- Dice coefficient loss -------------------------------
# Recall
def recall(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred) + 1
    denominator = K.sum(y_true) + 1
    return intersection / denominator


# Precision
def precision(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred) + 1
    denominator = K.sum(y_pred) + 1
    return intersection / denominator


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred) + 1
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


# Dice coeff.loss
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# --------------------------------- (Focal) Tversky loss ----------------------------------------
def tversky_loss(beta, gamma):
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        denominator = K.sum(y_true * y_pred) + (1 - beta) * K.sum((1 - y_true) * y_pred) + beta * K.sum(y_true * (1 - y_pred))
        loss_value = (intersection + 1) / (denominator + 1)

        return 1.0 - K.pow(loss_value, (1/gamma))

    return loss


# ----------------------------------- Focal loss ------------------------------------------

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))   #TP
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))  #TN
        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return loss


# ------------------------------ Binary focal loss ------------------------------
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)

        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)

        # Calculate focal loss
        loss = weight * cross_entropy

        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed



# ------------------------------ CE + DL -----------------------------
def CE_DL_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
      y_pred = tf.math.sigmoid(y_pred)
      numerator = 2 * tf.reduce_sum(y_true * y_pred)
      denominator = tf.reduce_sum(y_true + y_pred)

      return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

# ------------------------ Improved dice loss (including negative region) -------------
def new_dice_loss(y_true, y_pred):
    smooth = 1.0
    v1 = 0.05
    v2 = 0.95
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    w = (y_true * (v2 - v1)) + v1
    y_true = w * (2.0 * y_true - 1.0)
    y_pred = w * (2.0 * y_pred - 1.0)

    intersection = K.sum(y_true * y_pred)
    denominator = K.sum(K.pow(y_true, 2)) + K.sum(K.pow(y_pred, 2))
    dice = 2.0 * (intersection + smooth) / (denominator + smooth)
    return 1.0 - dice