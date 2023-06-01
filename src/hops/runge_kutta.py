def integrate_RK4(y, t_index, dt, f):
    """
    Computes y_{n+1} from y_{n} using RK4 integration with the time step dt.
    Using this function one can solve differential equations of the form
    
    \frac{d}{dt} y = f(y, t)
    
    numerically.

    Parameters
    ----------
    y : np.ndarray
        state of the system, vector of dtype complex and length N 
    t_index : int
        current time index (can for example be an index into an array storing discrete time values).
        Important: (t_index + 1) corresponds to the time ts[t_index] + 0.5*dt.
    dt : float
        time step (dt == ts[i+2] - ts[i])
    f : function
        function that computes the right hand side of the differential equation, given as inputs
        the t_index and the current state vector y
        
    Returns
    -------
    np.ndarray 
        the state of the system after time step dt has passed
    """
    k1 = f(t_index, y)
    k2 = f(t_index + 1, y + 0.5*dt*k1)
    k3 = f(t_index + 1, y + 0.5*dt*k2)
    k4 = f(t_index + 2, y + dt*k3)
    return y + 1/6*(k1 + 2*k2 + 2*k3 + k4)*dt

def integrate_RK4_with_memory(y, m, t_index, dt, f):
    """
    Computes y_{n+1} and m_{n+1} from y_{n} and m_{n} using RK4 integration with the time step dt.
    Using this function one can solve differential equations of the form
    
    \frac{d}{dt} y = f_y(y, m, t)
    \frac{d}{dt} m = f_m(y, m, t)
    
    numerically.
    ------------------------------------
    Parameters:
    y : np.ndarray
        state of the system, vector of dtype complex and length N
    m : complex
        memory value that gets updated every time step as well
    t_index : int
        current time index (can for example be an index into an array storing discrete time values)
        Important: (t_index + 1) corresponds to the time ts[t_index] + 0.5*dt
    dt : float
        time step (dt == ts[i+2] - ts[i])
    f : function
        function that computes the right hand side of the differential equations, given as inputs
        the t_index, the current state vector y, and the current memory m
    ------------------------------------
    Returns:
    np.ndarray 
        the state of the system after time step dt has passed
    complex
        the memory value of the system after the time step dt has passed
    """
    k1_y, k1_m = f(t_index, y, m)
    k2_y, k2_m = f(t_index + 1, y + 0.5*dt*k1_y, m + 0.5*dt*k1_m)
    k3_y, k3_m = f(t_index + 1, y + 0.5*dt*k2_y, m + 0.5*dt*k2_m)
    k4_y, k4_m = f(t_index + 2, y + dt*k3_y, m + dt*k3_m)
    return y + 1/6*(k1_y + 2*k2_y + 2*k3_y + k4_y)*dt, m + 1/6*(k1_m + 2*k2_m + 2*k3_m + k4_m)*dt
