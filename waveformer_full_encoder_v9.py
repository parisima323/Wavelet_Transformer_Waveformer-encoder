import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from scipy.optimize import minimize
import math

# ----------------------------------------------------------------------
# 1. MODWT and iMODWT Implementations (Unchanged)
# ----------------------------------------------------------------------

def circular_convolution(x, h):
    """ Performs circular convolution of a 1D signal x with a filter h. """
    N = len(x)
    L = len(h)
    y = np.zeros(N)
    for n in range(N):
        sum_val = 0.0
        for l in range(L):
            idx = (n - l) % N
            sum_val += h[l] * x[idx]
        y[n] = sum_val
    return y

def modwt_step(V_j_minus_1, h_j, g_j):
    """ Performs one step of the MODWT decomposition. """
    W_j = circular_convolution(V_j_minus_1, h_j)
    V_j = circular_convolution(V_j_minus_1, g_j)
    return W_j, V_j

def get_modwt_filters(wavelet, j):
    """ Gets the MODWT filters (h_j, g_j) for a given wavelet and level j. """
    w = pywt.Wavelet(wavelet)
    h_dwt, g_dwt = w.dec_lo, w.dec_hi
    h_modwt = np.array(h_dwt) / np.sqrt(2)
    g_modwt = np.array(g_dwt) / np.sqrt(2)
    upsample_factor = 2**(j - 1)
    h_j = np.zeros(len(h_modwt) + (len(h_modwt) - 1) * (upsample_factor - 1))
    h_j[::upsample_factor] = h_modwt
    g_j = np.zeros(len(g_modwt) + (len(g_modwt) - 1) * (upsample_factor - 1))
    g_j[::upsample_factor] = g_modwt
    return h_j, g_j

def modwt(x, wavelet, n_levels):
    """ Maximal Overlap Discrete Wavelet Transform (MODWT) """
    V = x
    coeffs = []
    for j in range(1, n_levels + 1):
        h_j, g_j = get_modwt_filters(wavelet, j)
        W_j, V_j = modwt_step(V, h_j, g_j)
        coeffs.append(W_j)
        V = V_j
    coeffs.append(V)
    return coeffs

def imodwt_step(W_j, V_j, h_j, g_j):
    """ Performs one step of the inverse MODWT reconstruction. """
    h_j_inv = h_j[::-1]
    g_j_inv = g_j[::-1]
    V_j_minus_1 = circular_convolution(W_j, h_j_inv) + circular_convolution(V_j, g_j_inv)
    return V_j_minus_1

def imodwt(coeffs, wavelet):
    """ Inverse Maximal Overlap Discrete Wavelet Transform (iMODWT) """
    n_levels = len(coeffs) - 1
    V = coeffs[-1]
    for j in range(n_levels, 0, -1):
        W_j = coeffs[j-1]
        h_j, g_j = get_modwt_filters(wavelet, j)
        V = imodwt_step(W_j, V, h_j, g_j)
    return V

# ----------------------------------------------------------------------
# 2. Accurate Huber-Periodogram and Fisher's Test (Unchanged)
# ----------------------------------------------------------------------

def huber_loss(r, k=1.345):
    """ Huber loss function. """
    abs_r = np.abs(r)
    return np.where(abs_r <= k, 0.5 * r**2, k * abs_r - 0.5 * k**2)

def m_periodogram_objective(beta, signal, freq, k):
    """ Objective function for M-Periodogram (Huber Loss). """
    N = len(signal)
    t = np.arange(N)
    # The model is: signal = beta[0] * cos(...) + beta[1] * sin(...)
    residuals = signal - (beta[0] * np.cos(2 * np.pi * freq * t) + beta[1] * np.sin(2 * np.pi * freq * t))
    return np.sum(huber_loss(residuals, k))

def huber_periodogram(signal, k=1.345):
    """ M-Periodogram (Huber-Periodogram) based on minimizing Huber Loss. """
    N = len(signal)
    # Only consider the first half of the spectrum (excluding DC component)
    freqs = np.fft.rfftfreq(N, 1.0)[1:]
    Pxx = np.zeros(len(freqs))
    
    for i, f in enumerate(freqs):
        # Minimize Huber Loss to find the robust Fourier coefficients (beta_opt)
        res = minimize(m_periodogram_objective, x0=[0, 0], args=(signal, f, k), method='Nelder-Mead')
        beta_opt = res.x
        # The power is proportional to the square of the robust coefficients
        Pxx[i] = 0.5 * N * (beta_opt[0]**2 + beta_opt[1]**2)
        
    return Pxx

def fisher_g_test(periodogram_values, alpha=0.05):
    """ Calculates Fisher's g-statistic and performs the test. """
    if len(periodogram_values) == 0 or np.sum(periodogram_values) == 0:
        return False
    
    # Only consider the first half of the spectrum (excluding DC component)
    g_statistic = np.max(periodogram_values) / np.sum(periodogram_values)
    m = len(periodogram_values)
    
    # Calculate the p-value for Fisher's g-statistic
    p_value = sum(((-1)**(k-1)) * math.comb(m, k) * np.power(max(0, 1 - k * g_statistic), m - 1) for k in range(1, m + 1))
    
    return p_value < alpha

def huber_fisher_test(signal, alpha=0.05):
    """ Combines Huber-Periodogram and Fisher's g-test. """
    periodogram_values = huber_periodogram(signal)
    return fisher_g_test(periodogram_values, alpha)

# ----------------------------------------------------------------------
# 3. Wavelet Attention with Accurate Validation (Unchanged)
# ----------------------------------------------------------------------

class WaveletAttention(nn.Module):
    """ Wavelet Attention (WA) Mechanism with full MODWT, Validation, and iMODWT. """
    def __init__(self, d_model, n_levels, wavelet='db4', dropout=0.1):
        super(WaveletAttention, self).__init__()
        self.d_model, self.n_levels, self.wavelet = d_model, n_levels, wavelet
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def modwt_decomposition(self, x):
        """ Performs MODWT, Huber-Fisher validation, and iMODWT reconstruction. """
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(-1, seq_len).detach().cpu().numpy()
        all_wavelet_features = []
        
        for signal in x_reshaped:
            coeffs = modwt(signal, self.wavelet, self.n_levels)
            W_coeffs, V_J = coeffs[:-1], coeffs[-1]
            
            # Filter coefficients based on Huber-Fisher Test
            filtered_coeffs = [W_j if huber_fisher_test(W_j) else np.zeros_like(W_j) for W_j in W_coeffs]
            filtered_coeffs.append(V_J)
            
            W_features = imodwt(filtered_coeffs, self.wavelet)
            all_wavelet_features.append(W_features)
            
        wavelet_features_np = np.array(all_wavelet_features)
        wavelet_features = torch.tensor(wavelet_features_np, dtype=x.dtype, device=x.device)
        return wavelet_features.reshape(batch_size, d_model, seq_len).permute(0, 2, 1)

    def forward(self, x):
        Q, K, V = self.query_projection(x), self.key_projection(x), self.value_projection(x)
        W = self.modwt_decomposition(x)
        K_modulated = K + W
        scores = torch.matmul(Q, K_modulated.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attn), V)
        return self.out_projection(context), attn

# ----------------------------------------------------------------------
# 4. Level Module (Final Implementation based on Eq. 11)
# ----------------------------------------------------------------------

class LevelModule(nn.Module):
    """
    Level Module implementing the Level Smoothing logic (Eq. 11) from the Waveformer paper.
    
    L_{t}^{(n)} = \alpha * (L_{t-1}^{(n)} - \text{Linear}(S_{t}^{(n)})) + (1 - \alpha) * (L_{t-1}^{(n)} + \text{Linear}(T_{t}^{(n)}))
    """
    def __init__(self, d_model):
        super(LevelModule, self).__init__()
        # Learnable smoothing parameter alpha (element-wise)
        self.alpha = nn.Parameter(torch.ones(1, 1, d_model) * 0.1)
        
        # Linear layers for the Linear() terms in the formula
        self.linear_seasonal = nn.Linear(d_model, d_model)
        self.linear_trend = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend_init, seasonal_comp, trend_comp):
        # trend_init: L_{t-1}^{(n)} (Level/Trend from previous layer)
        # seasonal_comp: S_{t}^{(n)} (Output of Wavelet Attention)
        # trend_comp: T_{t}^{(n)} (Output of MH-ESA)
        
        # Ensure alpha is between 0 and 1 (using sigmoid for stability)
        alpha = torch.sigmoid(self.alpha)
        
        # Term 1: alpha * (L_{t-1}^{(n)} - Linear(S_{t}^{(n)}))
        term1 = alpha * (trend_init - self.linear_seasonal(seasonal_comp))
        
        # Term 2: (1 - alpha) * (L_{t-1}^{(n)} + Linear(T_{t}^{(n)}))
        term2 = (1 - alpha) * (trend_init + self.linear_trend(trend_comp))
        
        # L_{t}^{(n)} = Term 1 + Term 2
        trend_update = term1 + term2
        
        # Apply LayerNorm (as seen in the diagram/standard practice)
        return self.norm(trend_update)

# ----------------------------------------------------------------------
# 5. Full Encoder Architecture (Updated LevelModule usage)
# ----------------------------------------------------------------------

class MHESA(nn.Module):
    """ Multi-Head Extraction and Smoothing Attention (MH-ESA) for Trend Extraction. """
    def __init__(self, d_model, n_heads, kernel_size, dropout=0.1):
        super(MHESA, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.smoothing = MovingAvg(kernel_size=kernel_size, stride=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multi_head_attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return self.smoothing(x)

class WaveformerEncoderLayer(nn.Module):
    """ A single layer of the Waveformer Encoder, now including MH-ESA and Level Module. """
    def __init__(self, d_model, n_levels, n_heads, kernel_size, d_ff=None, dropout=0.1, wavelet='db4'):
        super(WaveformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.wavelet_attention = WaveletAttention(d_model, n_levels, wavelet, dropout)
        self.mhesa = MHESA(d_model, n_heads, kernel_size, dropout)
        self.level_module = LevelModule(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, trend_init):
        # 1. Seasonal Component Update (Wavelet Attention)
        attn_output, _ = self.wavelet_attention(x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. Trend Component Update (MH-ESA and Level Module)
        trend_extracted = self.mhesa(trend_init)
        # LevelModule now takes Seasonal (x) and Trend (trend_extracted) components
        trend_update = self.level_module(trend_init, x, trend_extracted)
        
        # 3. Seasonal Component Update (Feed-Forward)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, trend_update

# ----------------------------------------------------------------------
# 6. Remaining Components (Unchanged)
# ----------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        return self.avg(x.permute(0, 2, 1)).transpose(1, 2)

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend

class WaveformerEncoder(nn.Module):
    def __init__(self, c_in, d_model, n_layers, n_levels, n_heads, wavelet='db4', d_ff=None, dropout=0.1, decomp_kernel=25, mhesa_kernel=5):
        super(WaveformerEncoder, self).__init__()
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=dropout)
        self.decomposition = SeriesDecomposition(decomp_kernel)
        self.layers = nn.ModuleList([WaveformerEncoderLayer(d_model, n_levels, n_heads, mhesa_kernel, d_ff, dropout, wavelet) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_raw):
        x = self.embedding(x_raw)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_output, trend_update = seasonal_init, trend_init
        for layer in self.layers:
            seasonal_output, trend_update = layer(seasonal_output, trend_update)
        return self.norm(seasonal_output), trend_update

# ----------------------------------------------------------------------
# 7. Example Usage and Verification
# ----------------------------------------------------------------------

if __name__ == '__main__':
    c_in, d_model, seq_len, n_layers, n_levels, n_heads = 7, 512, 96, 3, 3, 8
    wavelet_type, batch_size, decomp_kernel, mhesa_kernel = 'db4', 32, 25, 5

    dummy_raw_input = torch.randn(batch_size, seq_len, c_in)
    full_encoder = WaveformerEncoder(c_in, d_model, n_layers, n_levels, n_heads, wavelet_type, decomp_kernel=decomp_kernel, mhesa_kernel=mhesa_kernel)
    
    print("--- Testing Final Waveformer Encoder (v9) ---")
    print(f"Raw Input shape: {dummy_raw_input.shape}")
    seasonal_output, trend_update = full_encoder(dummy_raw_input)
    print(f"Seasonal Output shape: {seasonal_output.shape}")
    print(f"Trend Output shape: {trend_update.shape}")
    
    assert seasonal_output.shape == (batch_size, seq_len, d_model)
    assert trend_update.shape == (batch_size, seq_len, d_model)
    print("Final Waveformer Encoder implementation check passed.")
