clear; close all; clc; 
%% 
load handel
v = y';
v(end) = [];
%{
plot((1:length(v))/Fs;,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');
%}

%{
p8 = audioplayer(v,Fs);
playblocking(p8);
%}

n = length(v);L = length(v)/Fs;
t = (1:length(v))/Fs;
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);

b = 0.1;
a = 100;
Sgt_spec = [];
tslide = 0:b:L;
for j=1:length(tslide)
%g=exp(-a*(t-tslide(j)).^2); % Gabor
%g= (1-a*(t-tslide(j)).^2).*exp(-a*(t-tslide(j)).^2/2);% MH
g= abs(t - tslide(j)) <= a/2; % Shannon 
Sg=g.*v; Sgt=fft(Sg);
Sgt_spec=[Sgt_spec; abs(fftshift(Sgt))];
end

pcolor(tslide,ks,Sgt_spec.'), shading interp
title('Step Function: a = 100, b = 0.1','Fontsize',16)
xlabel('time(s)')
ylabel('frequency')
colormap(hot)


%%
%piano
[ypia,Fspia]= audioread('music1.wav');
ypia = ypia';
tr_piano=length(ypia)/Fspia; % record time in seconds
n=length(ypia);L=tr_piano;
t=(1:length(ypia))/Fspia;
k=(2*pi/L)*[0:(n/2-1) -n/2:-1]; kspia=fftshift(k);

tslide_p = 0:0.1:L;
Sgtp_spec=[];

for j=1:length(tslide_p)
%gaussian
g=exp(-100*((t-tslide_p(j)).^2));
Sgp=g.*ypia;
Sgtp=fft(Sgp);
Sgtp_spec=[Sgtp_spec; abs(fftshift(Sgtp))];
end
pcolor(tslide_p,kspia/(2*pi),Sgtp_spec'), shading interp
xlabel('time(s)');ylabel('frequency(Hz)');title('Piano')
set(gca,'Ylim',[200 1000])
colormap(hot)


%%
%record
[yrec,Fsrec] = audioread('music2.wav');
yrec = yrec';
tr_rec=length(yrec)/Fsrec; % record time in seconds
n=length(yrec);L=tr_rec;
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ksr=fftshift(k);

tslide_r = 0:0.1:L;
Sgtr_spec=[];

for j=1:length(tslide_r)
%gaussian
g=exp(-100*((t-tslide_r(j)).^2));
Sgr=g.*yrec;Sgtr=fft(Sgr);
Sgtr_spec=[Sgtr_spec; abs(fftshift(Sgtr))];
end

pcolor(tslide_r,ksr/(2*pi),Sgtr_spec'), shading interp
xlabel('time(s)');ylabel('frequency(Hz)');title('Recorder')
ylim([200 1500])
colormap(hot)

