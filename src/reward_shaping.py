import numpy as np

PHI_WEIGHT = 5.0
SURVIVAL_REWARD = 1.0
PHI_DOT_WEIGHT = 1.0
ACTION_WEIGHT = 1.0
KILL_REWARD = -20000
PHI_WEIGHT_N = 5.0
LAMBDA = 0.001

def compute_reward(
    phi: float,
    phi_dot: float,
    setpoint: float,
    throttle: float,
    equilibrium: float,
    max_dangle: float,
    terminated: bool,
) -> float:
    """Calcula a recompensa modelada (shaping reward) para o ambiente do braço Roberta."""

    # Penalidade extrema para falha crítica (ex: ângulo do braço exceder os limites físicos permitidos).
    if terminated:
        return float(KILL_REWARD)

    # Cálculo de erro usando Norma L1 (valor absoluto). 
    # Isso garante um gradiente de correção constante, forçando o agente a corrigir 
    # os desvios mesmo quando está milimetricamente perto do alvo.
    # As métricas são normalizadas dividindo pelos limites máximos de amplitude (max_dangle).
    phi_dot_error = abs(phi_dot / (5 * max_dangle))
    phi_error = abs((phi - setpoint) / max_dangle)
    
    # Penalidade de esforço de controle (uso do motor). 
    # O peso reduzido (multiplicado por 0.1) garante que o agente foque na precisão  
    # posicional (chegar ao setpoint) em vez de apenas otimizar o consumo de energia.
    action_penalty = (ACTION_WEIGHT * 0.1) * abs(throttle - equilibrium)
    
    # Composição principal: O agente ganha uma base constante de sobrevivência e 
    # sofre decréscimos proporcionais aos erros de cinemática (ângulo e velocidade) 
    # e ao esforço da ação empregada.
    reward = -(
        PHI_DOT_WEIGHT * phi_dot_error
        + PHI_WEIGHT * phi_error
        + action_penalty
    ) + (
        PHI_WEIGHT_N * np.exp(-phi_error**2/LAMBDA)
        + SURVIVAL_REWARD
    )

    # Bônus de estado estacionário (Zona Alvo):
    # Condição de recompensa superdimensionada acionada apenas quando a precisão 
    # atinge menos de 2% de desvio, ancorando o braço no local exato do setpoint.
    if phi_error < 0.02: 
        reward += 5.0
    if phi_error < 0.005:
        reward += 5
        
    return float(reward)