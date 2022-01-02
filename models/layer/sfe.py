import torch.nn as nn

class ResUnit(nn.Module):
    def __init__(self, filter_num):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(filter_num),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = self.layer(x)
        out = residual + x
        return out

class Conv1ResUnitsConv2(nn.Module):
    def __init__(self, dconf, mconf):
        super().__init__()
        self.dconf = dconf
        self.mconf = mconf
        self.L = self.mconf.res_repetation
        self.Conv1 = nn.Conv2d(self.dconf.dim_flow, self.mconf.res_nbfilter, kernel_size=3, stride=1, padding=1)
        self.ResUnits = self._stack_resunits(self.mconf.res_nbfilter)
        self.SeLu = nn.SELU(inplace=True)
        self.Conv2 = nn.Conv2d(self.mconf.res_nbfilter, self.dconf.dim_flow, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features = self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w, out_features = self.mconf.transformer_dmodel)

    def _stack_resunits(self, filter_num):
        layers = []
        for i in range(0, self.L):
            layers.append(ResUnit(filter_num))
        return nn.Sequential(*layers)

    def forward(self, x, ext):
        x = self.Conv1(x)
        ext = ext.reshape(-1, self.mconf.res_nbfilter, self.dconf.dim_h, self.dconf.dim_w)
        out = x + ext
        out = self.ResUnits(out)
        out = self.SeLu(out)
        out = self.Conv2(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out
