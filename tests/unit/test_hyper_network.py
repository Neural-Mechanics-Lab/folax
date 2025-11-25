import pytest
import unittest
import os
import numpy as np
from fol.deep_neural_networks.nns import MLP,HyperNetwork
from fol.tools.usefull_functions import *
import jax

class TestHyperNetworks(unittest.TestCase):

    def test_hn_all_to_all(self):

        # first check all to all coupling
        with self.assertRaises(SystemExit):
            HyperNetwork(name="test_hypernetwork", 
                        modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                        synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[4,5,6]))

        with self.assertRaises(SystemExit):
            HyperNetwork(name="test_hypernetwork", 
                        modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                        synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3]),
                        coupling_settings={"coupled_variable":"weight"})

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[1,2,3],activation_settings={"type":"relu"}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3],activation_settings={"type":"sin"}))
        self.assertEqual(hyper_network.coupling_settings,{"coupled_variable":"shift",
                                                          "modulator_to_synthesizer_coupling_mode":"all_to_all"})
        self.assertEqual(hyper_network.total_num_biases,hyper_network.modulator_nn.total_num_biases+hyper_network.synthesizer_nn.total_num_biases)
        self.assertEqual(hyper_network.total_num_weights,hyper_network.modulator_nn.total_num_weights+hyper_network.synthesizer_nn.total_num_weights)
        self.assertEqual(hyper_network.fw_func,hyper_network.all_to_all_fw)

        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(1,20))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(10,3))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                                   np.array( [-0.05008760094642639, -0.0502968393266201, 0.029370781034231186, -0.049322620034217834, -0.04586054012179375, -0.019689474254846573, -0.0499572828412056, 0.03073960915207863, 0.02309989742934704, -0.04453772306442261] ), rtol=1e-5, atol=1e-6)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[1,2,3],activation_settings={"type":"relu"},skip_connections_settings={"active":True,"frequency":1}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[1,2,3],activation_settings={"type":"sin"},skip_connections_settings={"active":True,"frequency":1}))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                                   np.array( [-0.06464479118585587, 0.0014808910200372338, 0.01677718758583069, 0.08632665872573853, 0.032101988792419434, -0.006967228837311268, 0.05838199332356453, -0.0016861282056197524, -0.0013258749386295676, 0.04222153499722481] ), rtol=1e-5, atol=1e-6)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[2,2,2,2],activation_settings={"type":"relu"}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=1,hidden_layers=[2,2,2,2],activation_settings={"type":"sin"},skip_connections_settings={"active":True,"frequency":2}))
        
        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(), 
                                   np.array( [0.05745392665266991, -0.0007242548745125532, 0.008966343477368355, -0.05604708939790726, 0.0022205559071153402, 0.01883917860686779, -0.02079216204583645, 0.006951518822461367, 0.0036605263594537973, -0.039732251316308975] ), rtol=1e-5, atol=1e-6)

    def test_hn_last_to_all(self):
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"last_to_all"})
        self.assertEqual(hyper_network.modulator_nn.out_features,hyper_network.synthesizer_nn.total_num_biases-hyper_network.synthesizer_nn.out_features)
        self.assertEqual(hyper_network.fw_func,hyper_network.last_to_all_fw)

        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(1,20))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(3,3))

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.05131402239203453, -0.060301389545202255, 0.08443807065486908, -0.012814491987228394, -0.06959788501262665, -0.013126535341143608, -0.0115011902526021, -0.03213585540652275, -0.02539145015180111] )
                                   ,rtol=1e-5, atol=1e-6)
        
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,output_size=1,hidden_layers=[1,2,3],skip_connections_settings={"active":True,"frequency":2}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"last_to_all"})
        self.assertEqual(hyper_network.modulator_nn.out_features,hyper_network.synthesizer_nn.total_num_biases-hyper_network.synthesizer_nn.out_features)
        self.assertEqual(hyper_network.fw_func,hyper_network.last_to_all_fw)

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.05169450119137764, -0.020866356790065765, 0.069666787981987, 0.029222482815384865, 0.042269494384527206, -0.04068158194422722, 0.026434095576405525, 0.045258499681949615, -0.06289883702993393] )
                                   ,rtol=1e-5, atol=1e-6)
        
    def test_hn_one_modulator_per_synthesizer_layer(self):
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[5]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],skip_connections_settings={"active":True,"frequency":1}),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

        self.assertEqual(hyper_network.total_num_biases,48)
        self.assertEqual(hyper_network.total_num_weights,497)
        self.assertEqual(len(hyper_network.modulator_nns),3)
        self.assertEqual(hyper_network.modulator_nns[0].hidden_layers,[5])
        self.assertEqual(hyper_network.modulator_nns[0].in_features,20)
        self.assertEqual(hyper_network.modulator_nns[0].out_features,4)
        self.assertEqual(hyper_network.modulator_nns[0].skip_connections_settings,{"active":False,"frequency":1})
        self.assertEqual(hyper_network.modulator_nns[1].out_features,5)
        self.assertEqual(hyper_network.modulator_nns[2].out_features,6)
        self.assertEqual(hyper_network.fw_func,hyper_network.one_modulator_per_synthesizer_layer_fw)
        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(1,20))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(3,3))
        
        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.008463507518172264, 0.02936953492462635, 0.06820549815893173, 0.023692084476351738, 0.06468091905117035, 0.0033158911392092705, 0.04362775757908821, 0.10823079943656921, -0.02508261799812317] ),
                                   rtol=1e-5, atol=1e-6)

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,hidden_layers=[]),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6]),
                                    coupling_settings={"coupled_variable":"shift","modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

        self.assertEqual(hyper_network.total_num_biases,33)
        self.assertEqual(hyper_network.total_num_weights,380)
        self.assertEqual(len(hyper_network.modulator_nns),3)

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.0019369798246771097, -0.007256647106260061, 0.010852844454348087, 0.002259971806779504, 0.024501902982592583, 0.01460216287523508, 0.017710335552692413, 0.014407334849238396, 0.02802344039082527] ),
                                   rtol=1e-5, atol=1e-6)
        
    def test_hn_one_modulator_per_synthesizer_layer_with_fourier_feature_mapping(self):
        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,fourier_feature_settings={"active":True,"size":20,"learn_frequency":False}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],fourier_feature_settings={"active":True,"size":4,"learn_frequency":False}),
                                    coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})   

        self.assertEqual(hyper_network.CountTrainableParams(),733)     

        latent_vector = jax.random.normal(jax.random.PRNGKey(41), shape=(1,20))
        coord_matrix = jax.random.normal(jax.random.PRNGKey(41), shape=(3,3))    

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.025235909968614578, -0.018120964989066124, -0.01590644009411335, -0.031533729285001755, -0.015385366976261139, -0.0071960692293941975, -0.02646677754819393, -0.030212869867682457, -0.0148277897387743] ),
                                   rtol=1e-5, atol=1e-6)    

        hyper_network = HyperNetwork(name="test_hypernetwork", 
                                    modulator_nn=MLP(name="modulator_nn",input_size=20,fourier_feature_settings={"active":True,"size":20,"learn_frequency":True}),
                                    synthesizer_nn=MLP(name="synthesizer_nn",input_size=3,output_size=3,hidden_layers=[4,5,6],fourier_feature_settings={"active":True,"size":4,"learn_frequency":True}),
                                    coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})   

        self.assertEqual(hyper_network.CountTrainableParams(),1945)          

        np.testing.assert_allclose(np.array(hyper_network(latent_vector,coord_matrix)).flatten(),
                                   np.array( [-0.025235909968614578, -0.018120964989066124, -0.01590644009411335, -0.031533729285001755, -0.015385366976261139, -0.0071960692293941975, -0.02646677754819393, -0.030212869867682457, -0.0148277897387743] ),
                                   rtol=1e-5, atol=1e-6)                    

if __name__ == '__main__':
    unittest.main()