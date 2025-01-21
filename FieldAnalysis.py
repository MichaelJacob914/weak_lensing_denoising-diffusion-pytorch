#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Import Statements
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy.interpolate import make_interp_spline
from astropy.io import fits
from PIL import Image
from torchvision import transforms
from scipy.stats import gaussian_kde


# In[ ]:


class PowerSpectrumCalculator:
    def __init__(self, N_grid, L):
        self.set_map_properties(N_grid, L)
        self.set_fft_properties(N_grid, L)
            
    def set_map_properties(self, N_grid, L):
        self.N_grid   = N_grid
        self.L        = L
        self.Area     = self.L**2          # Area of the map  
        self.PIX_AREA = self.Area / self.N_grid**2

    def set_fft_properties(self, N_grid, L):
        kx = 2 * np.pi * np.fft.fftfreq(N_grid, d=L / N_grid)
        ky = 2 * np.pi * np.fft.fftfreq(N_grid, d=L / N_grid)

        self.N_Y = (N_grid//2 +1)
        
        # mesh of the 2D frequencies
        self.kx = np.tile(kx[:, None], (1, self.N_Y))
        self.ky = np.tile(ky[None, :self.N_Y], (N_grid, 1))
        self.k  = np.sqrt(self.kx**2 + self.ky**2)

        self.kmax = self.k.max()
        self.kmin = np.sort(self.k.flatten())[1]
        
        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)
        
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * np.fft.rfftn(x_map) / self.Area
        return np.array([Fx_complex.real, Fx_complex.imag])

    def get_k_bins(self, N_bins):
        return np.logspace(np.log10(self.kmin), np.log10(self.kmax), N_bins)
    
    def set_k_bins(self, N_bins):
        self.k_bins = self.get_k_bins(N_bins)

    def binned_Pk(self, delta1, delta2=None):
        cross_Pk_bins = []
        k_bin_centre = []
        delta_ell_1 = self.map2fourier(delta1)
        if delta2 is not None:
            delta_ell_2 = self.map2fourier(delta2)
        else:
            delta_ell_2 = delta_ell_1
        for i in range(len(self.k_bins) - 1):
            select_k = (self.k > self.k_bins[i]) & (self.k < self.k_bins[i+1]) & self.fourier_symm_mask
            k_bin_centre.append(np.mean(self.k[select_k]))
            # The factor of 2 needed because there are both real and imaginary modes in the l selection!
            cross_Pk = 2. * np.mean(delta_ell_1[:,select_k] * delta_ell_2[:,select_k]) * self.Area
            cross_Pk_bins.append(cross_Pk)


        return np.array(k_bin_centre), np.array(cross_Pk_bins)

    def binned_Pk_correlation(self, delta1, delta2):
        cross_Pk_bins = []
        k_bin_centre = []
        delta_ell_1 = self.map2fourier(delta1)

        delta_ell_2 = self.map2fourier(delta2)

        for i in range(len(self.k_bins) - 1):
            select_k = (self.k > self.k_bins[i]) & (self.k < self.k_bins[i+1]) & self.fourier_symm_mask
            k_bin_centre.append(np.mean(self.k[select_k]))
            # The factor of 2 needed because there are both real and imaginary modes in the l selection!
            cross_Pk = 2. * np.mean(delta_ell_1[:,select_k] * delta_ell_2[:,select_k]) * self.Area
            cross_Pk_bins.append(cross_Pk)

        return np.array(k_bin_centre), np.array(cross_Pk_bins)

class FieldCorrelations:
    """
    This class streamlines the process of calculating specific statistical metrics for two fields, 
    specifically the power spectrum, PDF, peak counts, and void counts.
    """

    def __init__(self, diffusion_sample, true_map, field_length,  plot_name, KS_inverse = None, comp_fields = None, mask=np.ones((256, 256), dtype=bool), is_tensor=True, normal_plot = True, comparison = False):
        self.diffusion_sample = diffusion_sample
        self.true_map = true_map
        self.plot_name = plot_name
        self.field_length = field_length
        self.mask = mask
        self.is_tensor = is_tensor
        self.comparison = comparison
        self.comp_fields = comp_fields
        if(KS_inverse != None):
            self.plotThird = True
            self.KS_inverse = KS_inverse
            self.plot_comparison()
        elif(normal_plot):
            self.plotThird = False
            self.plot_comparison()
        else:
            self.plot_comparison_2()
        

    def plot_comparison(self):
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))  # 3x2 grid of subplots

        min_val = min(self.diffusion_sample.min(), self.true_map.min())
        max_val = max(self.diffusion_sample.max(), self.true_map.max())
        # 1. Plot the two fields side by side
        im1 = axs[0, 0].imshow(self.diffusion_sample, cmap='viridis')#, vmin=min_val, vmax=max_val)
        axs[0, 0].set_title('Diffusion Sample')

        if(self.comparison):
            im2 = axs[0, 1].imshow(self.true_map, cmap='viridis')#, vmin=min_val, vmax=max_val)
            axs[0, 1].set_title('True Map')
        # Plot true_map with the shared color scale
        else: 
            self.plot_ratio(axs[0, 1])
            axs[0, 1].set_title('Power Spectrum Ratio')


        # Add a single colorbar for both images
        #cbar = fig.colorbar(im1, ax=axs[0, :], orientation='horizontal', fraction=0.046, pad=0.04)
        #cbar.set_label('Intensity')
        
        # 2. Plot the power spectrum of the two fields (stacked)
        self.plot_power_spectrum(axs[1, 0])
        if(self.comp_fields != None): 
           axs[1,0] = self.plot_median_power_spectrum(axs[1,0], self.comp_fields, label = 'Diffusion Sample Median Power Spectrum')

        # 3. Plot the cross-correlation between the two fields
        self.plot_normalized_correlation(axs[1, 1])

        # 4. Plot the PDF of the two fields (stacked)
        self.plot_PDF(axs[2, 0])
        if(self.comp_fields != None): 
           axs[2,0] = self.plot_median_pdf(axs[2,0], self.comp_fields)

        # 5. Plot KDE Peaks on the remaining axis
        self.plot_kde_peaks(axs[2, 1])
        if(self.comp_fields != None): 
           axs[2,1] = self.plot_median_kde_peaks(axs[2,1], self.comp_fields)

        plt.tight_layout()
        plt.savefig(self.plot_name + '.png')  # Save the plot as a PNG file

    def plot_comparison_2(self):
        fig = plt.figure(figsize=(30, 30))
        # Set up gridspec with equal width for columns and height ratios for rows
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.4, wspace=0.4)
    
        # Row 1: Three fields
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.diffusion_sample, cmap='viridis')
        ax1.set_title('Diffusion Sample 1', loc='center')
    
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.true_map, cmap='viridis')
        ax2.set_title('Diffusion Sample 2', loc='center')
    
        ax3 = fig.add_subplot(gs[0, 2])
        if self.field_3 is not None:
            ax3.imshow(self.field_3, cmap='viridis')
        ax3.set_title('True Map', loc='center')
    
        # Row 2: Power spectrum and Correlation Function (Equal width for columns 0 and 1)
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_power_spectrum(ax4)
        ax4.set_title('Power Spectrum', loc='center')
    
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_normalized_correlation(ax5)
        ax5.set_title('Correlation Function', loc='center')
    
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')  # Empty subplot for alignment
    
        # Row 3: PDF and KDE Peaks (Equal width for columns 0 and 1)
        ax7 = fig.add_subplot(gs[2, 0])
        self.plot_PDF(ax7)
        ax7.set_title('PDF', loc='center')
    
        ax8 = fig.add_subplot(gs[2, 1])
        self.plot_kde_peaks(ax8)
        ax8.set_title('KDE Peaks', loc='center')
        
    
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')  # Empty subplot for alignment
    
        plt.tight_layout()
        #plt.savefig('field_correlations.png')  # Save the plot as a PNG file
        plt.show()
    

    def get_neighbor_maps(self, flat_map):
        n, m = flat_map.shape
        neighbor_maps = []

        # Define the shifts for neighbors (8 directions)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right

        for dx, dy in shifts:
            shifted_map = np.zeros_like(flat_map)
            for i in range(n):
                for j in range(m):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < m:
                        shifted_map[i, j] = flat_map[ni, nj]
                    else:
                        shifted_map[i, j] = 0  # Or some other boundary value
            neighbor_maps.append(shifted_map)

        return np.array(neighbor_maps)

    def get_kappa_peaks(self, flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        max_neighbor_map = np.max(neighbor_maps, axis=0)
        select_peaks = (flat_map > max_neighbor_map) & mask
        return flat_map[select_peaks]

    def get_kappa_voids(self, flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        min_neighbor_map = np.min(neighbor_maps, axis=0)
        select_voids = (flat_map < min_neighbor_map) & mask
        return flat_map[select_voids]

    def plot_kde_peaks(self, ax=None):
        diffusion_sample = self.diffusion_sample.detach().cpu().numpy()
        true_map = self.true_map.detach().cpu().numpy()
        peaks_1 = self.get_kappa_peaks(diffusion_sample, self.mask)
        peaks_2 = self.get_kappa_peaks(true_map, self.mask)

        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure and axis if none are provided

        sns.kdeplot(peaks_1, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='b', label='Diffusion Sample')
        sns.kdeplot(peaks_2, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='r', label='True Map')
        if(self.plotThird):
            peaks_3 = self.get_kappa_peaks(self.KS_inverse.detach().cpu().numpy(), self.mask)
            sns.kdeplot(peaks_3, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='orange', label='Kaiser Inversion')
        
        ax.set_title('Kappa Peaks', fontsize=18)
        ax.set_xlabel('Peak Value', fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        return ax

    def plot_PDF(self, ax=None):
        data_1 = self.diffusion_sample.flatten()
        data_2 = self.true_map.flatten()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        sns.kdeplot(data_1, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='b', label='Diffusion Sample')
        sns.kdeplot(data_2, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='r', label='True Map')
        if(self.plotThird):
            data_3 = self.KS_inverse.cpu().flatten()
            sns.kdeplot(data_3, fill=None, alpha=0.7, linestyle='dashed', ax=ax, color='orange', label='Kaiser Inversion')

        ax.set_title('Smooth Probability Density Function (PDF)', fontsize=18)
        ax.set_xlabel('Values', fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        

        return ax  # Return the axis object

    def plot_power_spectrum(self, ax=None, N_grid=256):
        L = 3.5  # Length of the square map (assuming units)

        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)

        k_bin_centre, Pk_1 = psc.binned_Pk(self.diffusion_sample)
        _, Pk_2 = psc.binned_Pk(self.true_map)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.loglog(k_bin_centre, Pk_1, marker='o', linestyle='-', color='b', label='Diffusion Sample')
        ax.loglog(k_bin_centre, Pk_2, marker='o', linestyle='-', color='r', label='True Map')
        if(self.plotThird):
            _, Pk_3 = psc.binned_Pk(self.KS_inverse.cpu())
            ax.loglog(k_bin_centre, Pk_3, marker='o', linestyle='-', color='orange', label='Kaiser Inversion')

        ax.set_title('Power Spectrum of Flat Sky Map', fontsize=18)
        ax.set_xlabel('k (Spatial Frequency)', fontsize=15)
        ax.set_ylabel('k * P(k) (Power Spectrum)', fontsize=15)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()

        return ax  # Return the axis object

    def plot_normalized_correlation(self, ax=None, N_grid=256):
        L = 3.5  # Length of the square map (assuming units)

        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)

        k_bin_centre, Pk_11 = psc.binned_Pk(self.diffusion_sample)
        _, Pk_22 = psc.binned_Pk(self.true_map)

        _, Pk_12 = psc.binned_Pk(self.diffusion_sample, self.true_map)

        correlation_coefficient = Pk_12 / np.sqrt(Pk_11 * Pk_22)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_bin_centre, correlation_coefficient, marker='o', linestyle='-', color='b')
        if(self.plotThird):
            _, Pk_33 = psc.binned_Pk(self.KS_inverse.cpu())
            _, Pk_23 = psc.binned_Pk(self.KS_inverse.cpu(), self.true_map)
            correlation_coefficient = Pk_23 / np.sqrt(Pk_22 * Pk_33)
            ax.loglog(k_bin_centre, correlation_coefficient, marker='o', linestyle='-', color='orange', label='Kaiser Inversion')
        ax.set_title('Normalized Cross-Power Spectrum (Correlation Coefficient)', fontsize=18)
        ax.set_xlabel('k (Spatial Frequency)', fontsize=15)
        ax.set_ylabel('Correlation Coefficient', fontsize=15)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        return ax

    def plot_ratio(self, ax=None, N_grid=256):
        L = 3.5  # Length of the square map (assuming units)

        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)

        k_bin_centre, Pk_11 = psc.binned_Pk(self.diffusion_sample)
        _, Pk_22 = psc.binned_Pk(self.true_map)

        ratio = Pk_22 / Pk_11

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_bin_centre, ratio, marker='o', linestyle='-', color='b', label = 'Diffusion Sample Ratio')
        if(self.plotThird):
            _, Pk_33 = psc.binned_Pk(self.KS_inverse.cpu())
            ratio = Pk_33/Pk_22
            ax.loglog(k_bin_centre, ratio, marker='o', linestyle='-', color='orange', label='Kaiser Inversion Ratio')
        ax.set_title('Normalized Cross-Power Spectrum (Correlation Coefficient)', fontsize=18)
        ax.set_xlabel('k (Spatial Frequency)', fontsize=15)
        ax.set_ylabel('Correlation Coefficient', fontsize=15)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        return ax

    def plot_median_kde_peaks(self, ax, fields):

        def plot_kde_peaks(ax, fields, label, color, confidence_interval = False):
            all_peaks = []
            for field in fields:
                peaks = self.get_kappa_peaks(field, self.mask)
                all_peaks.append(peaks)
            all_peaks = np.concatenate(all_peaks)
        
            all_kdes = []
            x_grid = np.linspace(np.min(all_peaks), np.max(all_peaks), 1000)
            for field in fields:
                peaks = self.get_kappa_peaks(field, self.mask)
                if len(peaks) > 0:
                    kde = gaussian_kde(peaks)
                    all_kdes.append(kde(x_grid))
        
            all_kdes = np.array(all_kdes)
            median_kde = np.median(all_kdes, axis=0)
            lower_bound = np.percentile(all_kdes, 2.5, axis=0)
            upper_bound = np.percentile(all_kdes, 97.5, axis=0)
            
            ax.plot(x_grid, median_kde, label=label, color=color)
            if(confidence_interval):
                ax.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
            #ax.set_title('Median Peak Counts with 95% Confidence Interval', fontsize=18)
            ax.set_xlabel('Peak Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

    
        # Plot the KDE peaks for the main fields
        temp_fields = []
        for i in fields: 
            temp_fields.append(i.detach().cpu().numpy())
        plot_kde_peaks(ax, temp_fields, 'Diffusion Sample Prior Median', 'purple', confidence_interval = True)
    
        return ax

    def plot_median_power_spectrum(self, ax, fields, label, color = 'purple'):
        L = 3.5
        N_grid = 256
        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)
        
        all_power_spectra = []
        for field in fields:
            k_bin_centre, cross_Pk_bins = psc.binned_Pk(field)
            all_power_spectra.append(cross_Pk_bins)

        all_power_spectra = np.array(all_power_spectra)
        median_power_spectrum = np.median(all_power_spectra, axis=0)
        lower_bound = np.percentile(all_power_spectra, 2.5, axis=0)
        upper_bound = np.percentile(all_power_spectra, 97.5, axis=0)
        
        ax.plot(k_bin_centre, median_power_spectrum, marker='o', linestyle='-', color=color, label=label)
        ax.fill_between(k_bin_centre, lower_bound, upper_bound, color='gray', alpha=0.3)
        ax.set_xlabel('k (Spatial Frequency)', fontsize=15)
        ax.set_ylabel('P(k) (Power Spectrum)', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        return ax

    def plot_median_pdf(self, ax, fields):
    
        def plot_pdf(ax, fields, label, color = 'blue', confidence_interval = False):
            flattened_fields = [field.flatten() for field in fields]
            all_kdes = []
            x_grid = np.linspace(np.min(flattened_fields), np.max(flattened_fields), 1000)
            for field in flattened_fields:
                kde = gaussian_kde(field)
                all_kdes.append(kde(x_grid))
            all_kdes = np.array(all_kdes)
            median_kde = np.median(all_kdes, axis=0)
            
            ax.plot(x_grid, median_kde, label=label, color=color)
            if(confidence_interval):
                lower_bound = np.percentile(all_kdes, 2.5, axis=0)
                upper_bound = np.percentile(all_kdes, 97.5, axis=0)
                ax.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
            ax.set_xlabel('Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()


        plot_pdf(ax, fields, 'Diffusion Sample Prior Median', 'purple', confidence_interval = True)

        return ax
