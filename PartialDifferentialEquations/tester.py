import torch

def calculate_operation( r: torch.Tensor): # 5 * (1 + log(x))
    # Account for the case where r = 0. In those cases, return 0
    r[r == 0] = 1
    return 5 * (1 + torch.log(r))

def calculate_operation_2(r: torch.Tensor, x: torch.Tensor): # 5 * (1 + log(x))
    r[r == 0] = 1
    return 5 * (1 + torch.log(r)) + x

def gaussian(r: torch.Tensor, radius: float):
    return torch.exp(r**2)

def gausian_ver_2(r: torch.Tensor, b: torch.Tensor):
    return torch.exp(r**2) + b

x = torch.Tensor([[0, 1, 2, 0, 3, 1],
                  [2, 0, 0, 0, 0, 1]])

b = torch.Tensor([[1, 1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2, 2]])

torch.set_printoptions(precision=2, sci_mode=False)

print('Guass 1 Before:', x)

result_g_1 = gaussian(x, 1)
print('Guass 1 After:', result_g_1)

result_g_2 = gausian_ver_2(x, b)
print('Guass 2 After:', result_g_2)

print('\n\n Operation 1 Before:', x)
result_op_1 = calculate_operation(x)
print('Operation 1 After:', result_op_1)

result_op_2 = calculate_operation_2(x, b)
print('Operation 2 After:', result_op_2)