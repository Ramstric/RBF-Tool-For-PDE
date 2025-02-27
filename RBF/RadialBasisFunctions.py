import torch

# Polyharmonic spline
# LaTex: r^{2n} \log(r)
def polyharmonic_spline(r: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        return r**k * torch.log(r)
    else:
        return r**k

def polyharmonic_spline_x(r: torch.Tensor, x: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        return r**(k - 2) * (x + ( (k * x * torch.log(r**2))/2 ))
    else:

        return r**(k - 2) * x

def polyharmonic_spline_t(r: torch.Tensor, t: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        return r**(k - 2) * (t + ( (k * t * torch.log(r**2))/2 ))
    else:
        return r**(k - 2) * t

def polyharmonic_spline_xx(r: torch.Tensor, x: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        #return (x**2 * r**(k-4))*( (4*k - 2*k*torch.log(r**2) + k**2*torch.log(r**2))/2 ) + (r**(k-2))( 1 + (k/2) * torch.log(r**2) )
        return (x**2 * r**(k-4))*( (4*k - 2*k*torch.log(r**2) + k**2*torch.log(r**2))/2 ) + (r**(k-2))*( 1 + (k/2) * torch.log(r**2) )
    else:
        return x**2 * r**(k - 4)

def polyharmonic_spline_derivative(r: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        return r**(k - 1) * (k * torch.log(r) + 1)
    else:
        return r**(k - 1)

def polyharmonic_spline_second_derivative(r: torch.Tensor, k: int):
    if k % 2 == 0:
        r[r == 0] = 1
        return r**(k - 2) * (k**2 * torch.log(r) + 3 * k)
    else:
        return r**(k - 2) * (k - 1)

# Gaussian RBF
# LaTex: \exp\left(-\frac{r^2}{2\sigma^2}\right)
def gaussian(r: torch.Tensor, radius: float):
    return torch.exp(- r**2 / (2 * radius**2))


def gaussian_derivative(r, radius):
    return - r * gaussian(r, radius) / (radius**2)


# Partial Derivatives
def gaussian_t(r: torch.Tensor, t: torch.Tensor, radius: float):
    return - t * gaussian(r, radius) / (radius**2)


def gaussian_xx(r: torch.Tensor, x: torch.Tensor, radius: float):
    return (x**2 - radius**2) / (radius**4 * torch.exp(r**2 / (2 * radius**2)))

def gaussian_tt(r: torch.Tensor, t: torch.Tensor, radius: float):
    return (t**2 - radius**2) / (radius**4 * torch.exp(r**2 / (2 * radius**2)))

# Multiquadric RBF
# LaTex: \sqrt{r^2 + \sigma^2}
def multiquadric(r: torch.Tensor, radius: float):
    return torch.sqrt(r**2 + radius**2)


def multiquadric_derivative(r: torch.Tensor, radius: float):
    return r / multiquadric(r, radius)


def multiquadric_second_derivative(r: torch.Tensor, radius: float):
    return radius**2 / (multiquadric(r, radius) * (r**2 + radius**2))


# Partial Derivatives
def multiquadric_t(r: torch.Tensor, t: torch.Tensor, radius: float):
    return t / multiquadric(r, radius)


def multiquadric_tt(r: torch.Tensor, x: torch.Tensor, radius: float):
    return (radius**2 + x**2) / (multiquadric(r, radius) * (r**2 + radius**2))


def multiquadric_xx(r: torch.Tensor, t: torch.Tensor, radius: float):
    return (radius**2 + t**2) / (multiquadric(r, radius) * (r**2 + radius**2))

