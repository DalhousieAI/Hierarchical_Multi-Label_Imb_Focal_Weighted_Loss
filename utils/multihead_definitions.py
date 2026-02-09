import os
import torch

from sklearn.metrics import average_precision_score
from utils.utils import node_f1

class MultiHeadedModel(
    torch.nn.Module
):
    def __init__(self, enc, heads):
        super(MultiHeadedModel, self).__init__()
        self.enc = enc
        self.heads = heads

    def forward(self, x):
        if self.enc is not None:
            x = self.enc(x)
        x = torch.cat([head(x).unsqueeze(dim=0) for head in self.heads], dim=0)
        return x
    
    def fit(
            self,
            train_loader, 
            optimizer, 
            conf_hml_loss, 
            device
            ):
        
        self.train()

        epoch_loss = 0
        epoch_acc = 0
        epoch_uncertainty = 0

        for i, (x, labels) in enumerate(train_loader):
            batch_size = x.size(0)
            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = self(x.float())

            loss, mean_prediction_probs, pred_probs, uncertainties = conf_hml_loss(output, labels)
            batch_uncertainty = uncertainties.mean()

            predicted = mean_prediction_probs > 0.5

            # Total number of labels
            total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train = (predicted == labels.byte()).sum()

            acc = correct_train / total_train

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()*batch_size
            epoch_acc += acc.item()*batch_size
            epoch_uncertainty += batch_uncertainty.item()*batch_size

            if i == 0:
                predicted_train = predicted
                constr_train = mean_prediction_probs

                y_train = labels
            else:
                predicted_train = torch.cat((predicted_train, predicted), dim=0)
                constr_train = torch.cat((constr_train, mean_prediction_probs), dim=0)
                
                y_train = torch.cat((y_train, labels), dim=0)

        # Average the loss and accuracy over the epoch
        total_samples = len(train_loader.dataset)

        mean_loss = epoch_loss / total_samples 
        mean_acc = epoch_acc / total_samples 
        mean_uncertainty = epoch_uncertainty / total_samples 
        
        # Calculate metrics
        optimistic_ap_score = average_precision_score(
            y_train.cpu().numpy(), constr_train.data.cpu().numpy(), average='micro'
        )
        ap_score = average_precision_score(
            y_train.cpu().numpy(), predicted_train.data.cpu().numpy(), average='micro'
        )
        f1, precision, recall, _, _, _ = node_f1(
            y_train.cpu().numpy(), predicted_train.data.cpu().numpy()
        )

        return mean_loss, mean_acc, mean_uncertainty, \
                optimistic_ap_score, ap_score, f1, precision, recall
    
    def test(
            self,
            eval_loader,
            test_eval_index,
            conf_hml_loss,
            seed,
            num_epochs,
            mix_enc_noise,
            num_heads,
            focal_k,
            focal_min,
            dataset_name,
            conf_method,
            write_to_file,
            device
    ):
        self.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(eval_loader):
                            
                x = x.to(device)
                y = y.to(device)

                output = self(x.float())
                loss, mean_prediction_probs, probs, uncertainties  = conf_hml_loss(output, y)

                predicted = mean_prediction_probs > 0.5
                
                # Total number of labels
                total = y.size(0) * y.size(1)
                # Total correct predictions
                correct = (predicted == y.byte()).sum()

                #Move output and label back to cpu to be processed by sklearn
                predicted = predicted.to('cpu')
                cpu_constrained_output = mean_prediction_probs.to('cpu')
                y = y.to('cpu')

                if i == 0:
                    predicted_test = predicted
                    constr_test = cpu_constrained_output

                    y_test = y
                else:
                    predicted_test = torch.cat((predicted_test, predicted), dim=0)
                    constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                    
                    y_test = torch.cat((y_test, y), dim =0)
            
            optimistic_ap_score = average_precision_score(
                y_test[:, test_eval_index], constr_test.data[:, test_eval_index], average='micro'
                )
            ap_score = average_precision_score(
                y_test[:, test_eval_index], predicted_test.data[:, test_eval_index], average='micro'
                )
            f1, precision, recall, _, _, _ = node_f1(
                y_test[:, test_eval_index], predicted_test.data[:, test_eval_index]
                )

            if write_to_file:
                full_filename = 'results/' + dataset_name + f'/{dataset_name}_conf.csv'

                if not os.path.exists(os.path.dirname(full_filename)):
                    os.makedirs(os.path.dirname(full_filename))
                # Open the file in append mode
                f = open(full_filename, 'a')

                # If the file is empty, write the header
                if os.stat(full_filename).st_size == 0:
                    f.write('conf_mode,seed,noise_factor,num_epochs,num_heads,focal_k,' + \
                            'focal_min,optimistic_ap_score,ap_score,' + \
                            'f1,precision,recall\n')
                f.write(
                    conf_method + ',' + str(seed) + ',' + str(mix_enc_noise) + ',' + 
                    str(num_epochs) + ',' + str(num_heads) + ',' +
                    str(focal_k) + ',' + str(focal_min) + ',' + str(optimistic_ap_score) + ',' +
                    str(ap_score) + ',' + str(f1) + ',' + str(precision) + ',' + str(recall) + '\n'
                )
                f.close()

                # Save constr_test.data to file
                pred_file_name = 'predictions/' + \
                        dataset_name + '_' + conf_method + \
                        '_s-' + str(seed)
                # replace "." with "-" in the file name
                pred_file_name = pred_file_name.replace(".", "-")
                pred_file_name = pred_file_name + '.pt'

                torch.save(constr_test.data, pred_file_name)
            else:
                return optimistic_ap_score, ap_score, f1, precision, recall
        
