
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n=400, change_point=200, mu1=0.0, mu2=2.0, sigma=1.0, seed=42):
    """
    Generate a synthetic time series of length `n`.
    The mean changes from mu1 to mu2 at `change_point`.
    """
    np.random.seed(seed)
    data = np.empty(n)
    data[:change_point] = np.random.normal(mu1, sigma, change_point)
    data[change_point:] = np.random.normal(mu2, sigma, n - change_point)
    return data

def cusum_detect(time_series, target_mean=0.0, k=0.5, h=5.0):
    """
    One-sided CUSUM for detecting upward shifts in the mean of `time_series`.
    
    Parameters
    ----------
    time_series : array-like
        The data to monitor
    target_mean : float
        Mean under the null hypothesis (no change)
    k : float
        Slack parameter (small offset from target_mean to reduce false alarms)
    h : float
        Detection threshold

    Returns
    -------
    S : np.array
        The CUSUM series
    change_points : list of int
        Indices where an alarm is triggered (S exceeds h)
    """
    n = len(time_series)
    S = np.zeros(n)
    change_points = []
    
    for t in range(1, n):
        # Deviation from target_mean minus the slack
        g_t = (time_series[t] - target_mean) - k
        # Standard CUSUM update
        S[t] = max(0, S[t-1] + g_t)
        
        if S[t] > h:
            # We raise an alarm: a potential upward shift
            change_points.append(t)
            # Optionally reset after detection to look for subsequent changes
            S[t] = 0
    
    return S, change_points

def demo_cusum():
    # Generate synthetic data
    n = 400
    change_point = 200
    data = generate_synthetic_data(n=n, change_point=change_point, mu1=0.0, mu2=2.0, sigma=1.0)

    # Apply CUSUM detection
    # Suppose we assume the 'normal' mean is 0.0
    S, change_points = cusum_detect(data, target_mean=0.0, k=0.5, h=5.0)
    
    # Plot
    plt.figure()
    plt.plot(data, label="Time Series")
    plt.title("Synthetic Time Series with Mean Shift at t={}".format(change_point))
    plt.xlabel("Time")
    plt.ylabel("Value")
    for cp in change_points:
        plt.axvline(cp, linestyle="--", label=f"Detected Alarm @ t={cp}")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(S, label="CUSUM Statistic")
    plt.title("CUSUM Statistic (One-Sided)")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    for cp in change_points:
        plt.axvline(cp, linestyle="--", label=f"Detected Alarm @ t={cp}")
    plt.legend()
    plt.show()

# ipython -i -m crptmidfreq.res.demo_cusum
if __name__ == "__main__":
    demo_cusum()
