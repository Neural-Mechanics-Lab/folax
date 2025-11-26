import pytest
import unittest
import os
import numpy as np
from fol.deep_neural_networks.nns import MLP
from fol.tools.usefull_functions import *
import jax
from flax import nnx

class TestMLP(unittest.TestCase):

    def test_siren_mlp(self):

        relu_mlp = MLP(name="relu_mlp",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[])
        self.assertEqual(relu_mlp.total_num_biases,1)
        self.assertEqual(relu_mlp.total_num_weights,3)

        relu_mlp = MLP(name="relu_mlp",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[2,2,2],
                     activation_settings={"type":"relu"})
        self.assertEqual(relu_mlp.total_num_biases,7)
        self.assertEqual(relu_mlp.total_num_weights,16)
        self.assertEqual(relu_mlp.activation_settings,{"type":"relu",
                                                       "prediction_gain":30,
                                                       "initialization_gain":1})
        self.assertEqual(relu_mlp.skip_connections_settings,{"active":False,"frequency":1})
        self.assertEqual(relu_mlp.act_func,jax.jax.nn.relu)
        self.assertEqual(relu_mlp.act_func_gain,1)
        self.assertEqual(relu_mlp.fw_func,relu_mlp.Forward)

        siren_mlp = MLP(name="siren",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[2,2,2],
                     activation_settings={"type":"sin",
                                          "prediction_gain":50,
                                          "initialization_gain":3},
                    skip_connections_settings={"active":True,"frequency":1})

        self.assertEqual(siren_mlp.total_num_biases,7)
        self.assertEqual(siren_mlp.total_num_weights,31)
        self.assertEqual(siren_mlp.activation_settings,{"type":"sin",
                                                        "prediction_gain":50,
                                                        "initialization_gain":3})
        self.assertEqual(siren_mlp.skip_connections_settings,{"active":True,"frequency":1})

        skip_siren_mlp = MLP(name="siren",
                     input_size=3,
                     output_size=2,
                     hidden_layers=[4,5,6,7],
                     activation_settings={"type":"sin",
                                          "prediction_gain":50,
                                          "initialization_gain":3},
                    skip_connections_settings={"active":True,"frequency":2})
        self.assertEqual(skip_siren_mlp.activation_settings,{"type":"sin",
                                                        "prediction_gain":50,
                                                        "initialization_gain":3})
        self.assertEqual(skip_siren_mlp.skip_connections_settings,{"active":True,"frequency":2})
        self.assertEqual(skip_siren_mlp.fw_func,skip_siren_mlp.ForwardSkip)
        self.assertEqual(skip_siren_mlp.total_num_biases,24)
        self.assertEqual(skip_siren_mlp.total_num_weights,142)
        self.assertEqual(skip_siren_mlp.nn_params[0][0].shape,(3,4))
        self.assertEqual(skip_siren_mlp.nn_params[1][0].shape,(4,5))
        self.assertEqual(skip_siren_mlp.nn_params[2][0].shape,(8,6))
        self.assertEqual(skip_siren_mlp.nn_params[3][0].shape,(6,7))
        self.assertEqual(skip_siren_mlp.nn_params[4][0].shape,(10,2))
        self.assertEqual(skip_siren_mlp.act_func,jax.numpy.sin)
        self.assertEqual(skip_siren_mlp.act_func_gain,50)

        nn_weights = [[[0.3896358013153076, -0.20457863807678223, 0.10382795333862305, -0.03369593620300293], 
                       [-0.6291625499725342, 0.06200003623962402, 0.3838648796081543, 0.48191046714782715], 
                       [-0.5835270881652832, -0.04785919189453125, 0.918410062789917, -0.4763765335083008]], 
                       [[0.004807222168892622, 0.026269085705280304, -0.047563664615154266, -0.020860940217971802, -0.0279169213026762], 
                        [0.07262720912694931, 0.023500608280301094, -0.05051074177026749, 0.02577456273138523, 0.028336791321635246], 
                        [-0.03215799853205681, 0.03722599148750305, 0.03011622279882431, -0.02489561401307583, -0.05042368546128273], 
                        [-0.044283635914325714, 0.020966410636901855, 0.02130838669836521, 0.015012252144515514, 0.05704798921942711]], 
                        [[-0.04292110726237297, 0.00666320463642478, -0.008229531347751617, -0.0019105438841506839, 0.013228489086031914, 0.022079581394791603], 
                         [-0.013813565485179424, 0.04273204505443573, 0.0006284484989009798, 0.01284101139754057, 0.0509902685880661, -0.048802345991134644], 
                         [0.04568653926253319, 0.016063185408711433, -0.0059873322024941444, 0.037280187010765076, 0.040686193853616714, 0.010358808562159538], 
                         [0.04439824819564819, -0.01892087422311306, -0.03403225541114807, -0.014587951824069023, 0.05185958743095398, -0.027729609981179237], 
                         [-0.012638705782592297, -0.045719172805547714, -0.0014665863709524274, 0.028479911386966705, -0.047335658222436905, 0.00704545434564352], 
                         [-0.04714442789554596, 0.047577038407325745, 0.019525734707713127, 0.019632164388895035, -0.012506271712481976, -0.032610032707452774], 
                         [0.013821456581354141, -0.03486763313412666, -0.03134452551603317, -0.03796743229031563, 0.04603566229343414, 0.036358438432216644], 
                         [-0.004331324249505997, -0.020489541813731194, -0.024847254157066345, -0.025675518438220024, 0.04924103617668152, -0.017624804750084877]], 
                         [[-0.05773934721946716, -0.019452380016446114, -0.030279649421572685, 0.011287965811789036, 0.05469025671482086, 0.03780931979417801, -0.04335426539182663], 
                          [-0.047612328082323074, -0.052022892981767654, 0.03429960086941719, -0.021513190120458603, -0.00683125015348196, -0.042258452624082565, -0.050663724541664124], 
                          [-0.05786493048071861, -0.012448839843273163, 0.0019723318982869387, -0.024941110983490944, 0.05164632573723793, -0.015154924243688583, 0.05143096297979355], 
                          [0.05403445288538933, -0.020768364891409874, 0.02295048162341118, -0.02637392468750477, 0.02510770782828331, 0.011287336237728596, -0.005256385542452335], 
                          [-0.04979241266846657, 0.0008906793664209545, 0.04764857515692711, 0.02714783139526844, -0.0015913438983261585, 0.009481115266680717, -0.043192435055971146], 
                          [-0.03981562703847885, 0.032268304377794266, 0.025338420644402504, 0.026119079440832138, -0.022632107138633728, 0.039654962718486786, 0.040751438587903976]], 
                          [[-0.014073711819946766, 0.008406516164541245], [0.0015401648124679923, -0.008104035630822182], [-0.004929383751004934, 0.007463841233402491], 
                           [-0.007268957793712616, 0.008026511408388615], [0.010711515322327614, 0.0058107865042984486], [-0.013294502161443233, -0.009576050564646721], 
                           [0.00673277024179697, -0.011852793395519257], [0.0032879412174224854, 0.014849449507892132], [-0.011729137040674686, -0.0039137122221291065], 
                           [-0.0145353302359581, -0.014690360054373741]]]
        
        for i in range(len(skip_siren_mlp.nn_params)):
            skip_siren_mlp.nn_params[i] = nnx.data((nnx.Param(jnp.array(nn_weights[i])),skip_siren_mlp.nn_params[i][1]))

        np.testing.assert_allclose(np.array(skip_siren_mlp(jnp.array([[1,2,3],[4,5,6],[7,8,9]]))).flatten(), 
                                    np.array([-0.06585833,-0.032813,-0.127911,-0.02603725,-0.20289356,-0.07241035]), rtol=1e-5, atol=1e-6)
        
        # now test fourier feature mapping
        fourier_feature_mlp = MLP(name="ff_mlp",
                                  input_size=2,
                                  output_size=1,
                                  hidden_layers=[5,5,5,5],
                                  activation_settings={"type":"relu"},
                                  fourier_feature_settings={"active":True,"size":5,"frequency_scale":1.0,"learn_frequency":False})    

        np.testing.assert_allclose(np.array(fourier_feature_mlp.B).flatten(), 
                                   np.array( [0.07520543038845062, 0.1719243973493576, 0.841088593006134, -0.5837331414222717, -0.11038121581077576, -1.5184602737426758, 1.1545977592468262, 0.25870752334594727, -0.045513253659009933, -0.3583025634288788] ), rtol=1e-5, atol=1e-6)    
        
        np.testing.assert_allclose(np.array(fourier_feature_mlp(jax.random.normal(jax.random.PRNGKey(42),(10,2)))).flatten(), 
                                   np.array( [-0.626466691493988, -0.19476424157619476, -0.027989299967885017, -0.04137006029486656, 0.0, -0.17605000734329224, 0.0, -0.10535581409931183, -0.1355840414762497, -0.040048614144325256] ), rtol=1e-5, atol=1e-6)

        self.assertEqual(fourier_feature_mlp.CountTrainableParams(),151)

        trainable_fourier_feature_mlp = MLP(name="ff_mlp",
                                  input_size=2,
                                  output_size=1,
                                  hidden_layers=[5,5,5,5],
                                  activation_settings={"type":"relu"},
                                  fourier_feature_settings={"active":True,"size":5,"frequency_scale":1.0,"learn_frequency":True})
        self.assertEqual(trainable_fourier_feature_mlp.CountTrainableParams(),161)         


if __name__ == '__main__':
    unittest.main()