from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.signal as ss
import librosa
import sys
import os
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import pad_sequence_input
from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.postprocessor.postprocessor import Postprocessor
from art.attacks.evasion import FastGradientMethod
import tensorflow as tf

from art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
)
import tensorflow.compat.v1 as tf1

if TYPE_CHECKING:
    # pylint: disable=C0412
    from tensorflow.compat.v1 import Tensor
    from tensorflow.compat.v1 import Session
    from torch import Tensor as PTensor

    from art.utils import SPEECH_RECOGNIZER_TYPE, TENSORFLOWV2_ESTIMATOR_TYPE, CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)


class AsrImperceptibleAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "masker",
        "eps",
        "learning_rate_1",
        "max_iter_1",
        "alpha",
        "learning_rate_2",
        "max_iter_2",
        "batch_size",
        "loss_theta_min",
        "decrease_factor_eps",
        "num_iter_decrease_eps",
        "increase_factor_alpha",
        "num_iter_increase_alpha",
        "decrease_factor_alpha",
        "num_iter_decrease_alpha",
    ]

    _estimator_requirements = (NeuralNetworkMixin, LossGradientsMixin, BaseEstimator, SpeechRecognizerMixin)

    def __init__(
            self,
            estimator,
            masker: "PsychoacousticMasker",
            eps: float = 2000.0,
            learning_rate_1: float = 100.0,
            max_iter_1: int = 1000,
            alpha: float = 0.05,
            learning_rate_2: float = 1.0,
            max_iter_2: int = 4000,
            loss_theta_min: float = 0.05,
            decrease_factor_eps: float = 0.8,
            num_iter_decrease_eps: int = 10,
            increase_factor_alpha: float = 1.2,
            num_iter_increase_alpha: int = 20,
            decrease_factor_alpha: float = 0.8,
            num_iter_decrease_alpha: int = 50,
            batch_size: int = 1,
    ) -> None:
        """
        Create an instance of the :class:`.ImperceptibleASR`.
        The default parameters assume that audio input is in `int16` range. If using normalized audio input, parameters
        `eps` and `learning_rate_{1,2}` need to be scaled with a factor `2^-15`
        :param estimator: A trained speech recognition estimator.
        :param masker: A Psychoacoustic masker.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate_1: Learning rate for stage 1 of attack.
        :param max_iter_1: Number of iterations for stage 1 of attack.
        :param alpha: Initial alpha value for balancing stage 2 loss.
        :param learning_rate_2: Learning rate for stage 2 of attack.
        :param max_iter_2: Number of iterations for stage 2 of attack.
        :param loss_theta_min: If imperceptible loss reaches minimum, stop early. Works best with `batch_size=1`.
        :param decrease_factor_eps: Decrease factor for epsilon (Paper default: 0.8).
        :param num_iter_decrease_eps: Iterations after which to decrease epsilon if attack succeeds (Paper default: 10).
        :param increase_factor_alpha: Increase factor for alpha (Paper default: 1.2).
        :param num_iter_increase_alpha: Iterations after which to increase alpha if attack succeeds (Paper default: 20).
        :param decrease_factor_alpha: Decrease factor for alpha (Paper default: 0.8).
        :param num_iter_decrease_alpha: Iterations after which to decrease alpha if attack fails (Paper default: 50).
        :param batch_size: Batch size.
        """

        # Super initialization
        super().__init__(estimator=estimator)
        self.masker = masker
        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self.alpha = alpha
        self.learning_rate_2 = learning_rate_2
        self.max_iter_2 = max_iter_2
        self._targeted = True
        self.batch_size = batch_size
        self.loss_theta_min = loss_theta_min
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        self._check_params()

        # init some aliases
        self._window_size = masker.window_size
        self._hop_size = masker.hop_size
        self._sample_rate = masker.sample_rate

        self._framework: Optional[str] = None

        if isinstance(self.estimator, TensorFlowV2Estimator):
            import tensorflow.compat.v1 as tf1

            # set framework attribute
            self._framework = "tensorflow"

            # disable eager execution and use tensorflow.compat.v1 API, e.g. Lingvo uses TF2v1 AP
            # tf1.disable_eager_execution()

            # TensorFlow placeholders
            self._delta = 0
            self._power_spectral_density_maximum_tf = 0
            self._masking_threshold_tf = 0
            # self._delta = tf1.placeholder(tf1.float32, shape=[None, None], name="art_delta")
            # self._power_spectral_density_maximum_tf = tf1.placeholder(tf1.float32, shape=[None], name="art_psd_max")
            # self._masking_threshold_tf = tf1.placeholder(
            #     tf1.float32, shape=[None, None, None], name="art_masking_threshold"
            # )
            # # TensorFlow loss gradient ops
            # self._loss_gradient_masking_threshold_op_tf = self._loss_gradient_masking_threshold_tf(
            #     self._delta, self._power_spectral_density_maximum_tf, self._masking_threshold_tf
            # )

        elif isinstance(self.estimator, PyTorchEstimator):
            # set framework attribute
            self._framework = "pytorch"

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate imperceptible, adversarial examples.
        :param x: An array with the original inputs to be attacked.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array holding the adversarial examples.
        """
        if y is None:
            raise ValueError("The target values `y` cannot be None. Please provide a `np.ndarray` of target labels.")

        nb_samples = x.shape[0]

        x_imperceptible = [None] * nb_samples

        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            # batch indices
            begin, end = m * self.batch_size, min((m + 1) * self.batch_size, nb_samples)

            # create batch of adversarial examples
            x_imperceptible[begin:end] = self._generate_batch(np.reshape(x[begin:end].transpose(), 22050), y[begin:end],
                                                              end-begin)

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object
        return np.array(x_imperceptible, dtype=dtype)

    def _generate_batch(self, x: np.ndarray, y: np.ndarray, batch_size) -> np.ndarray:
        """
        Create imperceptible, adversarial sample.
        This is a helper method that calls the methods to create an adversarial (`ImperceptibleASR._create_adversarial`)
        and imperceptible (`ImperceptibleASR._create_imperceptible`) example subsequently.
        """
        # create adversarial example
        x_adversarial = self._create_adversarial(x, y, batch_size)
        if self.max_iter_2 == 0:
            return x_adversarial

        # make adversarial example imperceptible
        x_imperceptible = self._create_imperceptible(x, x_adversarial, y)
        return x_imperceptible

    def _create_adversarial(self, x, y, batch_size) -> np.ndarray:
        """
        Create adversarial example with small perturbation that successfully deceives the estimator.
        The method implements the part of the paper by Qin et al. (2019) that is referred to as the first stage of the
        attack. The authors basically follow Carlini and Wagner (2018).
        | Paper link: https://arxiv.org/abs/1801.01944.
        :param x: An array with the original inputs to be attacked.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the adversarial outputs.
        """
        # print(np.argmax(y))
        # print(f"batch {batch_size}")

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object

        epsilon = [self.eps] * batch_size
        x_adversarial = x.copy()

        x_perturbed = x.copy()

        # Compute MFCC
        mfcc = librosa.feature.mfcc(x, self._sample_rate)
        mfcc2 = np.expand_dims(np.array(mfcc).flatten(), axis=0)

        for i in range(1, self.max_iter_1 + 1):
            # perform FGSM step for x
            gradients = self.estimator.loss_gradient(mfcc2, y, batch_mode=True)
            mfcc2 = mfcc2 - self.learning_rate_1 * np.array([np.sign(g) for g in gradients], dtype=dtype)
            # x_perturbed = x_perturbed - self.learning_rate_1 * np.array([np.sign(g) for g in gradients],
            #                                                            dtype=dtype)
            # Inverse MFCC
            mfcc2 = np.array(mfcc2).reshape(20, 44)
            x_perturbed = librosa.feature.inverse.mfcc_to_audio(mfcc=mfcc2)

            # Make sure dims match
            if x_perturbed.shape[0] > 22050:
                x_perturbed = x_perturbed[:22050]
            else:
                x_perturbed = np.pad(x_perturbed, (0, 22050 - x_perturbed.shape[0]))

            # clip perturbation
            perturbation = x_perturbed - x
            #perturbation = np.array([np.clip(p, -e, e) for p, e in zip(perturbation, epsilon)], dtype=dtype)  # TODO: aici e o problema
            perturbation = np.array([np.clip(np.array(perturbation), -e, e) for e in epsilon], dtype=dtype)

            # re-apply clipped perturbation to x
            x_perturbed = x + perturbation

            # Compute MFCC for perturbed audio
            x_perturbed = np.array(x_perturbed, dtype=np.float32)
            x_perturbed = x_perturbed.transpose()
            x_perturbed = np.reshape(x_perturbed, 22050)

            mfcc_perturbed = librosa.feature.mfcc(y=x_perturbed, sr=self._sample_rate)
            mfcc_perturbed = np.expand_dims(np.array(mfcc_perturbed).flatten(), axis=0)

            if i % self.num_iter_decrease_eps == 0:
                prediction = self.estimator.predict(mfcc_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    # validate adversarial target, i.e. f(x_perturbed)=y
                    print(f'prediction {np.argmax(prediction)} and label {np.argmax(y)}')
                    if np.argmax(prediction) == np.argmax(y):
                        # decrease max norm bound epsilon
                        perturbation_norm = np.max(np.abs(perturbation))
                        if epsilon[j] > perturbation_norm:
                            epsilon[j] = perturbation_norm
                        epsilon[j] *= self.decrease_factor_eps
                        # save current best adversarial example
                        x_adversarial = x_perturbed
                logger.info("Current iteration %s, epsilon %s", i, epsilon)

        # return perturbed x if no adversarial example found
        for j in range(batch_size + 1):
            print(f'j in create adv is {j}')
            print(f'x_adv is {x_adversarial}')
            print(f'x_pert is {x_perturbed}')
            if np.array_equal(x_perturbed, x):
                logger.critical("Adversarial attack stage 1 for x_%s was not successful", j)
                x_adversarial = x_perturbed

        return np.array(x_adversarial, dtype=dtype)

    def _create_imperceptible(self, x: np.ndarray, x_adversarial: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Create imperceptible, adversarial example with small perturbation.
        This method implements the part of the paper by Qin et al. (2019) that is described as the second stage of the
        attack. The resulting adversarial audio samples are able to successfully deceive the ASR estimator and are
        imperceptible to the human ear.
        :param x: An array with the original inputs to be attacked.
        :param x_adversarial: An array with the adversarial examples.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the imperceptible, adversarial outputs.
        """
        batch_size = 1 #x.shape[0]
        alpha_min = 0.0005
        print(x.shape)

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object

        early_stop = [False] * batch_size

        alpha = np.array([self.alpha] * batch_size, dtype=np.float32)
        loss_theta_previous = [np.inf] * batch_size
        x_imperceptible = [None] * batch_size
        # if inputs are *not* ragged, we can't multiply alpha * gradients_theta
        if x.ndim != 1:
            alpha = np.expand_dims(alpha, axis=-1)

        masking_threshold, psd_maximum = self._stabilized_threshold_and_psd_maximum(x)
        print(f'making_threshold = {masking_threshold.shape}')
        print(f'psd_max = {psd_maximum}')

        x_perturbed = x_adversarial.copy()
        # print(f'x_adv = {x_adversarial}')
        # print(f'x_perturbed in create imperceptible {x_perturbed}')
        # print(f'x in create imperceptible {x}')

        for i in range(1, self.max_iter_2 + 1):
            # get perturbation
            perturbation = x_perturbed - x

            x_perturbed = np.array(x_perturbed, dtype=float)
            # Compute MFCC for perturbed audio
            mfcc_perturbed = librosa.feature.mfcc(y=x_perturbed, sr=self._sample_rate)
            mfcc_perturbed = np.expand_dims(np.array(mfcc_perturbed).flatten(), axis=0)

            # get loss gradients of both losses
            gradients_net = self.estimator.loss_gradient(mfcc_perturbed, y, batch_mode=True)  # TODO
            gradients_theta, loss_theta = self._loss_gradient_masking_threshold(  # TODO
                perturbation, x, masking_threshold, psd_maximum
            )

            # check shapes match, otherwise unexpected errors can occur
            print(gradients_net.shape)
            assert gradients_net.shape == gradients_theta.shape

            # perform gradient descent steps
            mfcc_perturbed = mfcc_perturbed - self.learning_rate_2 * (gradients_net + alpha * gradients_theta)
            # x_perturbed = x_perturbed - self.learning_rate_2 * (gradients_net + alpha * gradients_theta)
            # Inverse MFCC
            mfcc2 = np.array(mfcc_perturbed).reshape(20, 44)
            x_perturbed = librosa.feature.inverse.mfcc_to_audio(mfcc=mfcc2)
            print(f'x shape = {x.shape} in create imperceptible')
            print(f'x perturbed shape = {x_perturbed.shape} in create imperceptible')

            # Make sure dims match
            if x_perturbed.shape[0] > 22050:
                x_perturbed = x_perturbed[:22050]
            else:
                x_perturbed = np.pad(x_perturbed, (0, 22050 - x_perturbed.shape[0]))

            if i % self.num_iter_increase_alpha == 0 or i % self.num_iter_decrease_alpha == 0:
                prediction = self.estimator.predict(mfcc_perturbed, batch_size=batch_size)  # TODO
                for j in range(batch_size):
                    # validate if adversarial target succeeds, i.e. f(x_perturbed)=y
                    if i % self.num_iter_increase_alpha == 0 and np.argmax(prediction) == np.argmax(y):
                        # increase alpha
                        alpha[j] *= self.increase_factor_alpha
                        # save current best imperceptible, adversarial example
                        if loss_theta[j] < loss_theta_previous[j]:
                            x_imperceptible = x_perturbed
                            loss_theta_previous[j] = loss_theta[j]

                    # validate if adversarial target fails, i.e. f(x_perturbed)!=y
                    if i % self.num_iter_decrease_alpha == 0 and np.argmax(prediction) != np.argmax(y):
                        # decrease alpha
                        alpha[j] = max(alpha[j] * self.decrease_factor_alpha, alpha_min)
                logger.info("Current iteration %s, alpha %s, loss theta %s", i, alpha, loss_theta)

            # note: avoids nan values in loss theta, which can occur when loss converges to zero.
            for j in range(batch_size):
                if loss_theta[j] < self.loss_theta_min and not early_stop[j]:
                    logger.warning(
                        "Batch sample %s reached minimum threshold of %s for theta loss.", j, self.loss_theta_min
                    )
                    early_stop[j] = True
            if all(early_stop):
                logger.warning(
                    "All batch samples reached minimum threshold for theta loss. Stopping early at iteration %s.", i
                )
                break

        # return perturbed x if no adversarial example found
        for j in range(batch_size):
            if x_imperceptible is None:
                logger.critical("Adversarial attack stage 2 for x_%s was not successful", j)
                x_imperceptible = x_perturbed

        return np.array(x_imperceptible, dtype=dtype)

    def _stabilized_threshold_and_psd_maximum(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return batch of stabilized masking thresholds and PSD maxima.
        :param x: An array with the original inputs to be attacked.
        :return: Tuple consisting of stabilized masking thresholds and PSD maxima.
        """
        masking_threshold = []
        psd_maximum = []
        # x.reshape(-1, 1)
        # x_padded, _ = pad_sequence_input(x)

        # for x_i in x:

        m_t, p_m = self.masker.calculate_threshold_and_psd_maximum(x)
        masking_threshold.append(m_t)
        psd_maximum.append(p_m)

        # stabilize imperceptible loss by canceling out the "10*log" term in power spectral density maximum and
        # masking threshold
        masking_threshold_stabilized = 10 ** (np.array(masking_threshold) * 0.1)
        psd_maximum_stabilized = 10 ** (np.array(psd_maximum) * 0.1)
        return masking_threshold_stabilized, psd_maximum_stabilized

    def _loss_gradient_masking_threshold(
            self,
            perturbation: np.ndarray,
            x: np.ndarray,
            masking_threshold_stabilized: np.ndarray,
            psd_maximum_stabilized: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute loss gradient of the global masking threshold w.r.t. the PSD approximate of the perturbation.
        The loss is defined as the hinge loss w.r.t. to the frequency masking threshold of the original audio input `x`
        and the normalized power spectral density estimate of the perturbation. In order to stabilize the optimization
        problem during back-propagation, the `10*log`-terms are canceled out.
        :param perturbation: Adversarial perturbation.
        :param x: An array with the original inputs to be attacked.
        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :return: Tuple consisting of the loss gradient, which has same shape as `perturbation`, and loss value.
        """
        # pad input
        # print(f'shape of perturbation is {perturbation.shape}')
        perturbation_padded, delta_mask = pad_sequence_input(perturbation)

        if self._framework == "tensorflow":
            # get loss gradients (TensorFlow)
            # feed_dict = {
            #     self._delta: perturbation_padded,
            #     self._power_spectral_density_maximum_tf: psd_maximum_stabilized,
            #     self._masking_threshold_tf: masking_threshold_stabilized,
            # }
            # pylint: disable=W0212
            gradients_padded, loss = self._loss_gradient_masking_threshold_tf(
                perturbation_padded, psd_maximum_stabilized, masking_threshold_stabilized)  #_sess.run(self._loss_gradient_masking_threshold_op_tf, feed_dict)  # TODO
        elif self._framework == "pytorch":
            # get loss gradients (TensorFlow)
            gradients_padded, loss = self._loss_gradient_masking_threshold_torch(
                perturbation_padded, psd_maximum_stabilized, masking_threshold_stabilized
            )
        else:
            raise NotImplementedError

        # undo padding, i.e. change gradients shape from (nb_samples, max_length) to (nb_samples)
        # lengths = delta_mask.sum(axis=1)
        # gradients = list()
        # for gradient_padded, length in zip(gradients_padded, lengths):
        #     gradient = gradient_padded[:length]
        #     gradients.append(gradient)

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object
        return np.array(gradients_padded, dtype=dtype), loss

    def _loss_gradient_masking_threshold_tf(
            self, perturbation: "Tensor", psd_maximum_stabilized: "Tensor", masking_threshold_stabilized: "Tensor"
    ) -> Union["Tensor", "Tensor"]:
        """
        Compute loss gradient of the masking threshold loss in TensorFlow.
        Note that the PSD maximum and masking threshold are required to be stabilized, i.e. have the `10*log10`-term
        canceled out. Following Qin et al (2019) this mitigates optimization instabilities.
        :param perturbation: Adversarial perturbation.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.
        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.
        """
        import tensorflow.compat.v1 as tf1

        # calculate approximate power spectral density
        psd_perturbation = self._approximate_power_spectral_density_tf(perturbation, psd_maximum_stabilized)
        print(psd_perturbation.shape)
        print(f'psd_max_stabilized = {psd_maximum_stabilized}')

        # calculate hinge loss
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.relu(psd_perturbation - masking_threshold_stabilized), axis=[1, 2], keepdims=False
            )
            tape.watch(loss)
        print(f'loss = {loss.shape}')
        # compute loss gradient
        perturbation = tf.convert_to_tensor(perturbation, dtype=tf.float32)
        loss_gradient = tape.gradient(loss, [perturbation])[0]
        print(loss_gradient)
        return loss_gradient, loss

    def _loss_gradient_masking_threshold_torch(
            self, perturbation: np.ndarray, psd_maximum_stabilized: np.ndarray, masking_threshold_stabilized: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Compute loss gradient of the masking threshold loss in PyTorch.
        See also `ImperceptibleASR._loss_gradient_masking_threshold_tf`.
        """
        import torch  # lgtm [py/import-and-import-from]

        # define tensors
        # pylint: disable=W0212
        perturbation_torch = torch.from_numpy(perturbation).to(self.estimator._device)
        masking_threshold_stabilized_torch = torch.from_numpy(masking_threshold_stabilized).to(self.estimator._device)
        psd_maximum_stabilized_torch = torch.from_numpy(psd_maximum_stabilized).to(self.estimator._device)

        # track gradient of perturbation
        perturbation_torch.requires_grad = True

        # calculate approximate power spectral density
        psd_perturbation = self._approximate_power_spectral_density_torch(
            perturbation_torch, psd_maximum_stabilized_torch
        )

        # calculate hinge loss
        loss = torch.mean(  # type: ignore
            torch.nn.functional.relu(psd_perturbation - masking_threshold_stabilized_torch), dim=(1, 2), keepdims=False
        )

        # compute loss gradient
        loss.sum().backward()
        loss_gradient = perturbation_torch.grad.cpu().numpy()
        loss_value = loss.detach().cpu().numpy()

        return loss_gradient, loss_value

    def _approximate_power_spectral_density_tf(
            self, perturbation: "Tensor", psd_maximum_stabilized: "Tensor"
    ) -> "Tensor":
        """
        Approximate the power spectral density for a perturbation `perturbation` in TensorFlow.
        Note that a stabilized PSD approximate is returned, where the `10*log10`-term has been canceled out.
        Following Qin et al (2019) this mitigates optimization instabilities.
        :param perturbation: Adversarial perturbation.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.
        """
        import tensorflow.compat.v1 as tf1

        # compute short-time Fourier transform (STFT)
        perturbation = tf.convert_to_tensor(perturbation, dtype=tf.float32)
        stft_matrix = tf1.signal.stft(perturbation, self._window_size, self._hop_size, fft_length=self._window_size)

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = np.sqrt(8.0 / 3.0)
        psd_matrix = tf1.square(tf1.abs(gain_factor * stft_matrix / self._window_size))

        # approximate normalized psd: psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        # print(psd_maximum_stabilized.shape)
        # print(psd_maximum_stabilized)
        # print(psd_matrix.shape)
        # print(psd_matrix)
        psd_matrix_approximated = tf1.pow(10.0, 9.6) / tf1.reshape(psd_maximum_stabilized, [-1, 1, 1]) * psd_matrix

        # return PSD matrix such that shape is (batch_size, window_size // 2 + 1, frame_length)
        return tf1.transpose(psd_matrix_approximated, [0, 2, 1])

    def _approximate_power_spectral_density_torch(
            self, perturbation: "PTensor", psd_maximum_stabilized: "PTensor"
    ) -> "PTensor":
        """
        Approximate the power spectral density for a perturbation `perturbation` in PyTorch.
        See also `ImperceptibleASR._approximate_power_spectral_density_tf`.
        """
        import torch  # lgtm [py/import-and-import-from]

        # compute short-time Fourier transform (STFT)
        # pylint: disable=W0212
        stft_matrix = torch.stft(
            perturbation,
            n_fft=self._window_size,
            hop_length=self._hop_size,
            win_length=self._window_size,
            center=False,
            window=torch.hann_window(self._window_size).to(self.estimator._device),
        ).to(self.estimator._device)

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = np.sqrt(8.0 / 3.0)
        stft_matrix_abs = torch.sqrt(torch.sum(torch.square(gain_factor * stft_matrix / self._window_size), -1))
        psd_matrix = torch.square(stft_matrix_abs)

        # approximate normalized psd: psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        psd_matrix_approximated = pow(10.0, 9.6) / psd_maximum_stabilized.reshape(-1, 1, 1) * psd_matrix

        # return PSD matrix such that shape is (batch_size, window_size // 2 + 1, frame_length)
        return psd_matrix_approximated

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if self.eps <= 0:
            raise ValueError("The perturbation max norm bound `eps` has to be positive.")

        if not isinstance(self.alpha, float):
            raise ValueError("The value of alpha must be of type float.")
        if self.alpha <= 0.0:
            raise ValueError("The value of alpha must be positive")

        if not isinstance(self.max_iter_1, int):
            raise ValueError("The maximum number of iterations for stage 1 must be of type int.")
        if self.max_iter_1 <= 0:
            raise ValueError("The maximum number of iterations for stage 1 must be greater than 0.")

        if not isinstance(self.max_iter_2, int):
            raise ValueError("The maximum number of iterations for stage 2 must be of type int.")
        if self.max_iter_2 < 0:
            raise ValueError("The maximum number of iterations for stage 2 must be non-negative.")

        if not isinstance(self.learning_rate_1, float):
            raise ValueError("The learning rate for stage 1 must be of type float.")
        if self.learning_rate_1 <= 0.0:
            raise ValueError("The learning rate for stage 1 must be greater than 0.0.")

        if not isinstance(self.learning_rate_2, float):
            raise ValueError("The learning rate for stage 2 must be of type float.")
        if self.learning_rate_2 <= 0.0:
            raise ValueError("The learning rate for stage 2 must be greater than 0.0.")

        if not isinstance(self.loss_theta_min, float):
            raise ValueError("The loss_theta_min threshold must be of type float.")

        if not isinstance(self.decrease_factor_eps, float):
            raise ValueError("The factor to decrease eps must be of type float.")
        if self.decrease_factor_eps <= 0.0:
            raise ValueError("The factor to decrease eps must be greater than 0.0.")

        if not isinstance(self.num_iter_decrease_eps, int):
            raise ValueError("The number of iterations must be of type int.")
        if self.num_iter_decrease_eps <= 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if not isinstance(self.num_iter_decrease_alpha, int):
            raise ValueError("The number of iterations must be of type int.")
        if self.num_iter_decrease_alpha <= 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if not isinstance(self.increase_factor_alpha, float):
            raise ValueError("The factor to increase alpha must be of type float.")
        if self.increase_factor_alpha <= 0.0:
            raise ValueError("The factor to increase alpha must be greater than 0.0.")

        if not isinstance(self.num_iter_increase_alpha, int):
            raise ValueError("The number of iterations must be of type int.")
        if self.num_iter_increase_alpha <= 0:
            raise ValueError("The number of iterations must be greater than 0.")

        if not isinstance(self.decrease_factor_alpha, float):
            raise ValueError("The factor to decrease alpha must be of type float.")
        if self.decrease_factor_alpha <= 0.0:
            raise ValueError("The factor to decrease alpha must be greater than 0.0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")


class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.
    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int = 2048, hop_size: int = 512, sample_rate: int = 16000) -> None:
        """
        Initialization.
        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.
        :param hop_size: Number of audio samples between adjacent STFT columns.
        :param sample_rate: Sampling frequency of audio inputs.
        """
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate

        # init some private properties for lazy loading
        self._fft_frequencies = None
        self._bark = None
        self._absolute_threshold_hearing: Optional[np.ndarray] = None

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the global masking threshold for an audio input and also return its maximum power spectral density.
        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also
        returns the maximum power spectral density (PSD) for each frame. Given an audio input, the following steps are
        performed:
        1. STFT analysis and sound pressure level normalization
        2. Identification and filtering of maskers
        3. Calculation of individual masking thresholds
        4. Calculation of global masking tresholds
        :param audio: Audio samples of shape `(length,)`.
        :return: Global masking thresholds of shape `(window_size // 2 + 1, frame_length)` and the PSD maximum for each
            frame of shape `(frame_length)`.
        """
        psd_matrix, psd_max = self.power_spectral_density(audio)
        print(f'psd_matrix = {psd_matrix.shape}')
        threshold = np.zeros_like(psd_matrix)
        for frame in range(psd_matrix.shape[1]):
            # apply methods for finding and filtering maskers
            maskers, masker_idx = self.filter_maskers(*self.find_maskers(psd_matrix[:, frame]))
            # apply methods for calculating global threshold
            threshold[:, frame] = self.calculate_global_threshold(
                self.calculate_individual_threshold(maskers, masker_idx)
            )
        return threshold, psd_max

    @property
    def window_size(self) -> int:
        """
        :return: Window size of the masker.
        """
        return self._window_size

    @property
    def hop_size(self) -> int:
        """
        :return: Hop size of the masker.
        """
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """
        :return: Sample rate of the masker.
        """
        return self._sample_rate

    @property
    def fft_frequencies(self) -> np.ndarray:
        """
        :return: Discrete fourier transform sample frequencies.
        """
        if self._fft_frequencies is None:
            self._fft_frequencies = np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> np.ndarray:
        """
        :return: Bark scale for discrete fourier transform sample frequencies.
        """
        if self._bark is None:
            self._bark = 13 * np.arctan(0.00076 * self.fft_frequencies) + 3.5 * np.arctan(
                np.square(self.fft_frequencies / 7500.0)
            )
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> np.ndarray:
        """
        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.
        """
        if self._absolute_threshold_hearing is None:
            # ATH applies only to frequency range 20Hz<=f<=20kHz
            # note: deviates from Qin et al. implementation by using the Hz range as valid domain
            valid_domain = np.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 2e4)
            freq = self.fft_frequencies[valid_domain] * 0.001

            # outside valid ATH domain, set values to -np.inf
            # note: This ensures that every possible masker in the bins <=20Hz is valid. As a consequence, the global
            # masking threshold formula will always return a value different to np.inf
            self._absolute_threshold_hearing = np.ones(valid_domain.shape) * -np.inf

            self._absolute_threshold_hearing[valid_domain] = (
                    3.64 * pow(freq, -0.8) - 6.5 * np.exp(-0.6 * np.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
            )
        return self._absolute_threshold_hearing

    def power_spectral_density(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral density matrix for an audio input.
        :param audio: Audio sample of shape `(length,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape
        `(frame_length)`.
        """
        import librosa

        # compute short-time Fourier transform (STFT)
        audio_float = audio.astype(np.float32)
        stft_params = {
            "n_fft": self.window_size,
            "hop_length": self.hop_size,
            "win_length": self.window_size,
            "window": ss.get_window("hann", self.window_size, fftbins=True),
            "center": False,
        }
        stft_matrix = librosa.core.stft(audio_float, **stft_params)

        # compute power spectral density (PSD)
        with np.errstate(divide="ignore"):
            gain_factor = np.sqrt(8.0 / 3.0)
            psd_matrix = 20 * np.log10(np.abs(gain_factor * stft_matrix / self.window_size))
            psd_matrix = psd_matrix.clip(min=-200)

        # normalize PSD at 96dB
        psd_matrix_max = np.max(psd_matrix)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix

        return psd_matrix_normalized, psd_matrix_max

    @staticmethod
    def find_maskers(psd_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify maskers.
        Possible maskers are local PSD maxima. Following Qin et al., all maskers are treated as tonal. Thus neglecting
        the nontonal type.
        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.
        :return: Possible PSD maskers and indices.
        """
        # identify maskers. For simplification it is assumed that all maskers are tonal (vs. nontonal).
        masker_idx = ss.argrelmax(psd_vector)[0]

        # smooth maskers with their direct neighbors
        psd_maskers = 10 * np.log10(np.sum([10 ** (psd_vector[masker_idx + i] / 10) for i in range(-1, 2)], axis=0))
        return psd_maskers, masker_idx

    def filter_maskers(self, maskers: np.ndarray, masker_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter maskers.
        First, discard all maskers that are below the absolute threshold of hearing. Second, reduce pairs of maskers
        that are within 0.5 bark distance of each other by keeping the larger masker.
        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Filtered PSD maskers and indices.
        """
        # filter on the absolute threshold of hearing
        # note: deviates from Qin et al. implementation by filtering first on ATH and only then on bark distance
        ath_condition = maskers > self.absolute_threshold_hearing[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]

        # filter on the bark distance
        bark_condition = np.ones(masker_idx.shape, dtype=bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            # find pairs of maskers that are within 0.5 bark distance of each other
            if self.bark[i] - self.bark[i_prev] < 0.5:
                # discard the smaller masker
                i_todelete, i_prev = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]

        return maskers, masker_idx

    def calculate_individual_threshold(self, maskers: np.ndarray, masker_idx: np.ndarray) -> np.ndarray:
        """
        Calculate individual masking threshold with frequency denoted at bark scale.
        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.
        """
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = np.zeros(masker_idx.shape + self.bark.shape)
        # TODO reduce for loop
        for k, (masker_j, masker) in enumerate(zip(masker_idx, maskers)):
            # critical band rate of the masker
            z_j = self.bark[masker_j]
            # distance maskees to masker in bark
            delta_z = self.bark - z_j
            # define two-slope spread function:
            #   if delta_z <= 0, spread_function = 27*delta_z
            #   if delta_z > 0, spread_function = [-27+0.37*max(PSD_masker-40,0]*delta_z
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]

            # calculate threshold
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function
        return threshold

    def calculate_global_threshold(self, individual_threshold):
        """
        Calculate global masking threshold.
        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.
        with np.errstate(divide="ignore"):
            return 10 * np.log10(
                np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10)
            )


class AsrEstimator(TensorFlowV2Estimator):
    estimator_params = TensorFlowV2Estimator.estimator_params + ["random_seed", "sess"]

    def __init__(self,
                 model,
                 clip_values: Optional["CLIP_VALUES_TYPE"] = None,
                 channels_first: Optional[bool] = None,
                 preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
                 postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
                 preprocessing: "PREPROCESSING_TYPE" = None,
                 random_seed: Optional[int] = None,
                 sess: Optional["Session"] = None):
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.random_seed = random_seed
        self._input_shape = None
        tf1.disable_eager_execution()
        # sys.path.append(self._LINGVO_CFG["path"])
        # tf1.flags.FLAGS(tuple(sys.argv[0]))
        self._x_padded: "Tensor" = tf1.placeholder(tf1.float32, shape=[None, None], name="art_x_padded")
        self._y_target: "Tensor" = tf1.placeholder(tf1.string, name="art_y_target")
        self._mask_frequency: "Tensor" = tf1.placeholder(tf1.float32, shape=[None, None, 80], name="art_mask_frequency")

    def predict(
        self, x: np.ndarray, batch_size: int = 128, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Perform batch-wise prediction for given inputs.

        :param x: Samples of shape `(nb_samples)` with values in range `[-32768, 32767]`. Note that it is allowable
                  that sequences in the batch could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param batch_size: Size of batches.
        :return: Array of predicted transcriptions of shape `(nb_samples)`. A possible example of a transcription
                 return is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        if x[0].ndim != 1:
            raise ValueError(
                "The LingvoASR estimator can only be used temporal data of type mono. Please remove any channel"
                "dimension."
            )
        # if inputs have 32-bit floating point wav format, the preprocessing argument is required
        is_normalized = max(map(max, np.abs(x))) <= 1.0  # type: ignore
        if is_normalized and self.preprocessing is None:
            raise ValueError(
                "The LingvoASR estimator requires input values in the range [-32768, 32767] or normalized input values"
                " with correct preprocessing argument (mean=0, stddev=1/normalization_factor)."
            )

        nb_samples = x.shape[0]
        assert nb_samples % batch_size == 0, "Number of samples must be divisible by batch_size"

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        y = list()
        nb_batches = int(np.ceil(nb_samples / float(batch_size)))
        for m in range(nb_batches):
            # batch indices
            begin, end = m * batch_size, min((m + 1) * batch_size, nb_samples)

            x_batch_padded, _, mask_frequency = self._pad_audio_input(x[begin:end])

            feed_dict = {
                self._x_padded: x_batch_padded,
                self._y_target: np.array(["DUMMY"] * batch_size),
                self._mask_frequency: mask_frequency,
            }
            # run prediction
            y_batch = self._sess.run(self._predict_batch_op, feed_dict)

            # extract and append transcription result
            y += y_batch["topk_decoded"][:, 0].tolist()

        y_decoded = [item.decode("utf-8").upper() for item in y]
        return np.array(y_decoded, dtype=str)


class DummyTensorFlowLingvoASR(SpeechRecognizerMixin, TensorFlowV2Estimator):
    """
    This class implements the task-specific Lingvo ASR model of Qin et al. (2019).

    The estimator uses a pre-trained model provided by Qin et al., which is trained using the Lingvo library and the
    LibriSpeech dataset.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html, https://arxiv.org/abs/1902.08295

    .. warning:: In order to calculate loss gradients, this estimator requires a user-patched Lingvo module. A patched
                 source file for the `lingvo.tasks.asr.decoder` module will be automatically applied. The original
                 source file can be found in `<PYTHON_SITE_PACKAGES>/lingvo/tasks/asr/decoder.py` and will be patched as
                 outlined in the following commit diff:
                 https://github.com/yaq007/lingvo/commit/414e035b2c60372de732c9d67db14d1003be6dd6

    The patched `decoder_patched.py` can be found in `ART_DATA_PATH/lingvo/asr`.

    Note: Run `python -m site` to obtain a list of possible candidates where to find the `<PYTHON_SITE_PACKAGES` folder.
    """

    estimator_params = TensorFlowV2Estimator.estimator_params + ["random_seed", "sess"]

    def __init__(
            self,
            model,
            clip_values: Optional["CLIP_VALUES_TYPE"] = None,
            channels_first: Optional[bool] = None,
            preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
            postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
            preprocessing: "PREPROCESSING_TYPE" = None,
            random_seed: Optional[int] = None,
            sess: Optional["Session"] = None,
    ):
        """
        Initialization.

        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
                used for data preprocessing. The first value will be subtracted from the input. The input will then
                be divided by the second one.
        :param random_seed: Specify a random seed.
        """
        import pkg_resources

        import tensorflow.compat.v1 as tf1

        # Super initialization
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.random_seed = random_seed

        self._input_shape = None

        # disable eager execution as Lingvo uses tensorflow.compat.v1 API
        tf1.disable_eager_execution()

        self._sess: "Session" = tf1.Session() if sess is None else sess
        self._loss_gradient_op: "Tensor" = self._loss_gradient(self._x_padded, self._y_target, self._mask_frequency)
        # init necessary local Lingvo ASR namespace and flags
        # sys.path.append(self._LINGVO_CFG["path"])
        # tf1.flags.FLAGS(tuple(sys.argv[0]))

        # placeholders
        self._x_padded: "Tensor" = tf1.placeholder(tf1.float32, shape=[None, None], name="art_x_padded")
        self._y_target: "Tensor" = tf1.placeholder(tf1.string, name="art_y_target")
        self._mask_frequency: "Tensor" = tf1.placeholder(tf1.float32, shape=[None, None, 80],
                                                         name="art_mask_frequency")


    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def sess(self) -> "Session":
        """
        Get current TensorFlow session.

        :return: The current TensorFlow session.
        """
        return self._sess

    @staticmethod
    def _check_and_download_file(uri: str, basename: str, *paths: str) -> str:
        """Check and download the file from given URI."""
        dir_path = os.path.join(*paths)
        file_path = os.path.join(dir_path, basename)
        if not os.path.isdir(dir_path):
            pass
            # make_directory(dir_path)
        if not os.path.isfile(file_path):
            logger.info("Could not find %s. Downloading it now...", basename)
            # get_file(basename, uri, path=dir_path)
        return file_path

    @staticmethod
    def _pad_audio_input(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply padding to a batch of audio samples such that it has shape of (batch_size, max_length)."""
        max_length = max(map(len, x))
        batch_size = x.shape[0]

        # calculate maximum frequency length
        assert max_length >= 480, "Maximum length of audio input must be at least 480."
        frequency_length = [((len(item) // 2 + 1) // 240 * 3) for item in x]
        max_frequency_length = max(frequency_length)

        x_padded = np.zeros((batch_size, max_length))
        x_mask = np.zeros((batch_size, max_length), dtype=bool)
        mask_frequency = np.zeros((batch_size, max_frequency_length, 80))

        for i, x_i in enumerate(x):
            x_padded[i, : len(x_i)] = x_i
            x_mask[i, : len(x_i)] = 1
            mask_frequency[i, : frequency_length[i], :] = 1
        return x_padded, x_mask, mask_frequency

    def loss_gradient(  # pylint: disable=W0221
            self, x: np.ndarray, y: np.ndarray, batch_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape `(nb_samples)`. Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param batch_mode: If `True` calculate gradient per batch or otherwise per sequence.
        :return: Loss gradients of the same shape as `x`.
        """
        # if inputs have 32-bit floating point wav format, the preprocessing argument is required
        is_normalized = max(map(max, np.abs(x))) <= 1.0  # type: ignore
        if is_normalized and self.preprocessing is None:
            raise ValueError(
                "The LingvoASR estimator requires input values in the range [-32768, 32767] or normalized input values"
                " with correct preprocessing argument (mean=0, stddev=1/normalization_factor)."
            )

        # Lingvo model works with lower case transcriptions
        y = np.array([y_i.lower() for y_i in y])

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        if batch_mode:
            gradients = self._loss_gradient_per_batch(x_preprocessed, y_preprocessed)
        else:
            gradients = self._loss_gradient_per_sequence(x_preprocessed, y_preprocessed)

        # Apply preprocessing gradients
        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

    def _loss_gradient_per_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x` per batch.
        """
        assert x.shape[0] == y.shape[0], "Number of samples in x and y differ."

        # pad input
        x_padded, mask, mask_frequency = self._pad_audio_input(x)

        # get loss gradients
        feed_dict = {
            self._x_padded: x_padded,
            self._y_target: y,
            self._mask_frequency: mask_frequency,
        }
        gradients_padded = self._sess.run(self._loss_gradient_op, feed_dict)

        # undo padding, i.e. change gradients shape from (nb_samples, max_length) to (nb_samples)
        lengths = mask.sum(axis=1)
        gradients = list()
        for gradient_padded, length in zip(gradients_padded, lengths):
            gradient = gradient_padded[:length]
            gradients.append(gradient)

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object
        return np.array(gradients, dtype=dtype)

    def _loss_gradient_per_sequence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x` per sequence.
        """
        assert x.shape[0] == y.shape[0], "Number of samples in x and y differ."

        # get frequency masks
        _, _, mask_frequency = self._pad_audio_input(x)

        # iterate over sequences
        gradients = list()
        for x_i, y_i, mask_frequency_i in zip(x, y, mask_frequency):
            # calculate frequency length for x_i
            frequency_length = (len(x_i) // 2 + 1) // 240 * 3

            feed_dict = {
                self._x_padded: np.expand_dims(x_i, 0),
                self._y_target: np.array([y_i]),
                self._mask_frequency: np.expand_dims(mask_frequency_i[:frequency_length], 0),
            }
            # get loss gradient
            gradient = self._sess.run(self._loss_gradient_op, feed_dict)  # type: ignore
            gradients.append(np.squeeze(gradient))

        # for ragged input, use np.object dtype
        dtype = np.float32 if x.ndim != 1 else np.object
        return np.array(gradients, dtype=dtype)

    def get_activations(
            self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
