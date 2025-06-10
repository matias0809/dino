norm = 'LN'  #'IN1d' 'LN''BN1d'
num_levels = 2
in_channels = 13 #ENDRE TIL MSI
embed_dims = 128  # embed_dims256 SKAL DETTE VÆRE SÅNN??
model = dict(
    num_feature_levels=num_levels,
    # dn_only_pos = False,
    backbone=dict(
        in_channels=in_channels,
        embed_dims=embed_dims,
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        num_levels=num_levels,
        norm_cfg=dict(type=norm),
        token_masking=True,  
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_levels=num_levels, num_points=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*8,  # 1024 for DeformDETR
                ffn_drop=0.0),
            norm_cfg=dict(type=norm),)),  # 0.1 for DeformDETR

     positional_encoding=dict(
        num_feats=embed_dims//2,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20)  # 10000 for DeformDETR
)