from common_imports import np, norm, gamma

class GreyNoise:
    def __init__(self, beta, S, K, r, sigma, t, T, Ns, Nmu):
        self.beta = beta
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.t = t
        self.T = T
        self.Ns = Ns
        self.Nmu = Nmu
        self.tau = T-t
    
    def grey_call_price(self):
        """calculate the option price using a grey noise model"""
        mu1=-self.sigma**2/2
        emu=0
        for n in range(self.Nmu):
            emu+= (-1)**n * gamma(1+2*n)*mu1**n / (np.math.factorial(n)* gamma(1+2*self.beta*n))
        mu=-np.log(emu)
        x=np.log(self.S/self.K) + self.r*self.tau
        c=-mu*self.tau**self.beta
        s=0
        for n in range (self.Ns+1):
            for m in range (1,self.Ns+1):
                s+= ((-1)**n)*((-x-mu*self.tau)**n)*(c**((m-n)/2))/(np.math.factorial(n)* gamma(1-self.beta*((n-m)/2)))
        return(s*self.K*np.exp(-self.r*self.tau)/2 )