clear; close all; clc;
load Testdata
L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% first row data with noise
Un(:,:,:)=reshape(Undata(1,:),n,n,n);
isosurface(X,Y,Z,abs(Un),0.4)
axis([-20 20 -20 20 -20 20]), grid on, drawnow
title('First row of data with noise','Fontsize',[15])
xlabel('x'),ylabel('y'),zlabel('z');

% determine the center frequency through averaging of the spectrum
Uavg2 = zeros(1,n^3);
Uavg = reshape(Uavg2,n,n,n);
for j=1:20
Un(:,:,:)=reshape(Undata(j,:),n,n,n);
Uavg = Uavg+fftn(Un);
end
Uavg = abs(fftshift(Uavg))/20;
[value1, index] = max(Uavg(:));
[xi,yi,zi]=ind2sub(size(Uavg),index);
kxc=Kx(xi,yi,zi);
kyc=Ky(xi,yi,zi);
kzc=Kz(xi,yi,zi);

% filter the data to denoise and determine the path 
gaussianf = exp(-0.2*((Kx-kxc).^2+(Ky-kyc).^2+(Kz-kzc).^2));
for j = 1:20
Un(:,:,:)=reshape(Undata(j,:),n,n,n);
Unft = fftshift(fftn(Un)).*gaussianf;
Unf = ifftn(ifftshift(Unft));
[value2,index2] = max(Unf(:));
[xj,yj,zj]=ind2sub(size(Unf),index2);
xp(j)=X(xj,yj,zj);
yp(j)=Y(xj,yj,zj);
zp(j)=Z(xj,yj,zj);
%close all,isosurface(X,Y,Z,abs(Unf),0.4)
%axis([-20 20 -20 20 -20 20]),
%xlabel('x'),ylabel('y'),zlabel('z'),grid on
end
plot3(xp,yp,zp,'k','Linewidth',[2])
hold on
plot3(xp(20),yp(20),zp(20),'or')
legend({'The Marble Path','Marble at the 20th measurement'},'Location','northwest')
title('Path of the Marble', 'Fontsize', [15])
xlabel('x'),ylabel('y'),zlabel('z'),grid on
% Destination of the path
[xp(20),yp(20),zp(20)]




