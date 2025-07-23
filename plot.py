import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, t


def plot_participants_mcc(participants = None, ci = 0.95):
    phone1, phone2 = gather_mcc_values(participants)
    phone1_mean, phone1_ci = matrix_to_mean_and_ci(phone1, ci)
    phone2_mean, phone2_ci = matrix_to_mean_and_ci(phone2, ci)
    l = f' ({participants.n_participants} Participants, CI={ci})'
    plt.figure(figsize=(10, 5))
    plt.axvline(x=2.5, color='grey', linestyle='--', linewidth=2, alpha=0.5,
        label='phone boundary')
    plt.plot(phone1_mean, label='Phone 1'+l, marker='o', color='blue')
    plt.fill_between(range(len(phone1_mean)), phone1_mean - phone1_ci, 
        phone1_mean + phone1_ci, alpha=0.2, color='blue')
    plt.plot(phone2_mean, label='Phone 2'+l, marker='x', color='red')
    plt.fill_between(range(len(phone2_mean)), phone2_mean - phone2_ci,
        phone2_mean + phone2_ci, alpha=0.2, color='red')
    plt.xticks(list(range(6)),list(range(1,7)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel('Diphone Gate')
    plt.ylabel('MCC')
    title = f'MCC for gated phone classification of diphones'
    plt.title(title)
    plt.show()
    



def gather_mcc_values(participants=None):
    if participants is None:
        participants = data.Participants()
    phone1 = np.zeros((participants.n_participants,6))
    phone2 = np.zeros((participants.n_participants,6))
    for row_index, participant in enumerate(participants.participants):
        for m in participant.matrices().matrices:
            col_index = index = m.gate - 1
            if m.diphone_position == 1:
                phone1[row_index, col_index] = m.mcc
            elif m.diphone_position == 2:
                phone2[row_index, col_index] = m.mcc
            else:
                raise ValueError(f"Invalid diphone position: {m.diphone_position}")
    return phone1, phone2

def matrix_to_mean_and_ci(matrix, ci = .95):
    """
    Calculate the mean and confidence interval of a matrix.
    
    Parameters:
    matrix (numpy.ndarray): Input matrix.
    
    Returns:
    tuple: (mean, ci) where mean is the average and ci is the 
    confidence interval.
    """
    alpha = 1 - ci
    mean = np.mean(matrix, axis=0)
    df = matrix.shape[0] - 1
    std_err = sem(matrix, axis=0)
    ci = std_err * t.ppf(1 - alpha / 2, df)
    return mean, ci
            
