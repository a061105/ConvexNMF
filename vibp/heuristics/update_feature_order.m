function model = update_feature_order( model , feature_order , param_set )

model.tau = model.tau(:,feature_order);
model.phi_mean = model.phi_mean(:,feature_order);
model.phi_cov = model.phi_cov(:,feature_order);
model.nu = model.nu(:,feature_order);
