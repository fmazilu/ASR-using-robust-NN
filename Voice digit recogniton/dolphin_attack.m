%fill in the path of normal audio file.
voice_file = 'data\seven\0b40aa8e_nohash_0.wav';

%read the signal from audio file.
[voice_signal, voice_samp_freq] = audioread(voice_file);

figure(1)
%plot original signal and its fft
t = linspace(0,length(voice_signal)/voice_samp_freq,length(voice_signal)) ;
subplot(221);
plot(t,voice_signal);
xlabel('Time [s]');
ylabel('Amplitude of Signal');
title('Original Signal-Amplitude');

N_fft = 2048;
Y_fft = fft(voice_signal,N_fft)/length(voice_signal);
f = voice_samp_freq/2*linspace(0,1,N_fft/2+1);
subplot(222);
plot(f,2*abs(Y_fft(1:N_fft/2+1)));
axis([0 voice_samp_freq/2 0 max(abs(Y_fft(1:N_fft/2+1)))])
xlabel('Frequency [Hz]');
ylabel('Magnitude of FFT');
title('Original Signal-FFT');


%pass through a low-pass filter to only keep frequencies below 7kHz
[b,a] = butter(10,[2*100/voice_samp_freq ,2*7000/voice_samp_freq],'bandpass');
% [b,a] = butter(10,2*7000/voice_samp_freq,'low');
voice_filter = filter(b,a,voice_signal(:,1));

%plot filtered signal and its fft
t = linspace(0,length(voice_filter)/voice_samp_freq,length(voice_filter)) ;
subplot(223);
plot(t,voice_filter);
xlabel('Time [s]');
ylabel('Amplitude of Signal');
title('Filtered Signal-Amplitude');

N_fft = 2048;
filtered_fft = fft(voice_filter,N_fft)/length(voice_filter);
f = voice_samp_freq/2*linspace(0,1,N_fft/2+1);
subplot(224);
plot(f,2*abs(filtered_fft(1:N_fft/2+1)));
axis tight;
xlabel('Frequency [Hz]');
ylabel('Magnitude of FFT');
title('Filtered Signal-FFT');


%upsample the signal with 192kHz sample rate
ultra_samp_freq = 192000;
voice_resamp = resample(voice_filter,ultra_samp_freq,voice_samp_freq);
voice_resamp = 1/max(abs(voice_resamp)) * voice_resamp;

%ultrasound modulation with 30kHz to obtain attack ultrasound
dt = 1/ultra_samp_freq;
len = size(voice_resamp,1);
t = (0:dt:(len - 1)*dt)';
carrier_freq = 30000;
ultrasound = voice_resamp.*cos(2*pi*carrier_freq*t) + 0.001*cos(2*pi*carrier_freq*t);
ultrasound = 1/max(abs(ultrasound)) * ultrasound;

figure(2)
%plot ultrasound signal and its fft
t = linspace(0,length(ultrasound)/ultra_samp_freq,length(ultrasound)) ;
subplot(211);
plot(t,ultrasound);
xlabel('Time [s]');
ylabel('Amplitude of Signal');
title('Modulated Signal-Amplitude');

N_fft = 2^nextpow2(length(ultrasound));
ultrasound_fft = fft(ultrasound,N_fft)/length(ultrasound);
f = ultra_samp_freq/2*linspace(0,1,N_fft/2+1);
subplot(212);
plot(f,2*abs(ultrasound_fft(1:N_fft/2+1)));
% axis([0 ultra_samp_freq/2 0 max(abs(ultrasound_fft(1:N_fft/2+1)))])
axis tight
xlabel('Frequency [Hz]');
ylabel('Magnitude of FFT');
title('Modulated Signal-FFT');

%write the attack signal into an audio file; fill in the path
ultrasound_file = 'dolphin_attack.wav';
audiowrite(ultrasound_file, ultrasound,ultra_samp_freq);