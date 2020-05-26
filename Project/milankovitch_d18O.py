import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt
from scipy import signal

EPS = 10**(-5)
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path()\
                                     + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['font.size']=16
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False

# define plotting subroutine
def makePlot(filename,x,y,xlab,ylab,title,legend=None,xlim=None,ylim=None,figsize=(14,6),semilog=False,annotations=None):
    fig = plt.figure(figsize=figsize)
    if not semilog:
        plt.plot(x,y)
    else:
        plt.semilogy(x,y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid()
    if legend:
        plt.legend(loc='best')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if annotations:
        for note in annotations:
            plt.annotate(note[0],xy=note[1])
    plt.show()
    fig.savefig(filename+".png", dpi=300)
    plt.close()

# define bandpass filtering subroutine
def butterworthFilter(x, lowf, highf, fs, order=5):
    low = lowf / (0.5*fs)
    high = highf / (0.5*fs)
    b,a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, x)

# load data
temp = np.loadtxt("benthic_data.txt",skiprows=1)
times = temp[:,0]
d18O = temp[:,1]
sterr = temp[:,2]

# subsample at period of 5 ka
times_sub = []
d18O_sub = []

for i,time in enumerate(times):
    # note that due to the variation in sampling frequency, between
    # indices of 600 and 1051, the signal was sampled at increment of 2 ka,
    # so we must average adjacent signal values in this range.
    if (time % 5.000 < EPS): # all other sampling increments divide 5 ka evenly.
        times_sub.append(time)
        d18O_sub.append(d18O[i])
    elif ((int(time) % 10 == 4) and (i > 600) and (i < 1051)):
        times_sub.append(0.5*(time+times[i+1]))
        d18O_sub.append(0.5*(d18O[i] + d18O[i+1]))
dt = times_sub[1]-times_sub[0]
times_sub = np.asarray(times_sub)
d18O_sub = np.asarray(d18O_sub)

# compute temperature
temp = 16.5 - 4.3*d18O_sub + 0.14*(d18O_sub**2)

# plot raw data
makePlot("temp_5ka_samp",times_sub,temp,"Time (ka)","Temperature (ºC)",\
         "Proxy for Average Deep Ocean Temperature over time, using the LR04"+\
         " Stack\n $f_s = (5 ka)^{-1} = 0.2 ka^{-1}$",\
         xlim=(times_sub[0],times_sub[-1]))

# detrend data
p = np.polyfit(times_sub, temp, 1)
trend = np.polyval(p, times_sub)
temp_detrended = temp - trend

# filter data
lowcut = 1./500.
highcut = 1./10.005
fs = 1/dt
temp_filt = butterworthFilter(temp_detrended, lowcut, highcut, fs)

# plot filtered and detrended temperature
makePlot("temp_filtered_5ka_samp",times_sub,temp_filt,"Time (ka)","Detrended "+\
         "Temperature ($\Delta$ºC)",\
         "Detrended and Filtered Temperature over time, using the LR04 Stack"+\
         "\n $f_s = 0.2 ka^{-1}$, Butterworth Bandpass Filter"+\
         " $[0.002 ka^{-1}, 0.1 ka^{-1}]$",\
         xlim=(times_sub[0],times_sub[-1]))

# compute DFT of filtered and detrended temperature
temp_filt_f_sh = np.fft.fftshift(np.fft.fft(temp_filt*dt))
freq_ax = np.fft.fftshift(np.fft.fftfreq(len(temp_filt_f_sh), dt))

# plot DFT amplitude of filtered and detrended temperature
makePlot("temp_fourier_5ka_samp_amp",freq_ax,np.abs(temp_filt_f_sh),\
         "Frequency ($ka^{-1}$)","Amplitude",\
         "DFT Amplitude of Filtered Temperature"+\
         "\n $f_s = 0.2 ka^{-1}$, Butterworth Bandpass Filter "+\
         "$[0.002 ka^{-1}, 0.1 ka^{-1}]$",
        xlim=(0,0.07))

# define convolution window
N = 25
hann = np.hanning(N)

# plot convolved DFT amplitude spectra
makePlot("temp_fourier_5ka_samp_amp_conv",freq_ax,\
        np.convolve(hann,np.abs(temp_filt_f_sh),'same'),\
         "Frequency ($ka^{-1}$)","Amplitude",\
         "DFT Amplitude of Smoothed, Filtered Temperature"+\
         "\n $f_s = 0.2 ka^{-1}$, Butterworth Bandpass Filter "+\
         "$[0.002 ka^{-1}, 0.1 ka^{-1}]$",
        xlim=(0,0.070),annotations=[["96.8 ka",(0.012,4200)],\
                                    ["40.6 ka",(0.022,3600)],\
                                    ["23.2 ka",(0.041,1600)]])

# finding the 41, 100, and 23 year periods
convolved = np.convolve(hann,np.abs(temp_filt_f_sh),'same')
from scipy.signal import argrelextrema
where = argrelextrema(convolved, np.greater) # compute local maxima
print("Printing amplitude values, frequencies, and period values at peaks:")
for wh in where[0]:
    if freq_ax[wh] >= 0.0:
        print("Amplitude: {:3f} \tFrequency: {:5f} \tPeriod: {:5f}".\
          format(convolved[wh],freq_ax[wh],1/freq_ax[wh]))
