class ModelConfigurationTaxiBJ:
    res_repetation = 12
    res_nbfilter = 16
    inter_extnn_inter_channels = 40
    inter_extnn_dropout = 0.5
    last_extnn_inter_channels = 40
    transformer_dmodel = 128
    transformer_nhead = 8
    transformer_dropout = 0.1
    transformer_nlayers = 2

    def show(self):
        print("res_repetation:", self.res_repetation)
        print("res_nbfilter:", self.res_nbfilter)
        print("inter_extnn_inter_channels:", self.inter_extnn_inter_channels)
        print("inter_extnn_dropout:", self.inter_extnn_dropout)
        print("last_extnn_inter_channels:", self.last_extnn_inter_channels)
        print("transformer_dmodel:", self.transformer_dmodel)
        print("transformer_nhead:", self.transformer_nhead)
        print("transformer_dropout:", self.transformer_dropout)
        print("transformer_nlayers:", self.transformer_nlayers)
