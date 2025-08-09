import torch
from config import Parameters
import torch.nn as nn
import torch.nn.functional as F

class point_pillars_loss(nn.Module):
    """
    Forward Input: loc, size, angle, clf, heading, loc0, size0, angle0, clf0, heading0
    """

    def __init__(self):
        super(point_pillars_loss, self).__init__()

        self.alpha = float(Parameters.alpha)
        self.gamma = float(Parameters.gamma)
        self.focal_weight = 1
        self.loc_weight = 2
        self.heading_weight = 0.2

        # loss function
        self.smoothL1 = nn.SmoothL1Loss()
        self.relu = nn.ReLU()
        # Binary cross entropy with logits for focal and direction loss
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_heading = nn.BCEWithLogitsLoss()


    
    def forward(self, loc, size, angle, clf, heading, loc0, size0, angle0, clf0, heading0):

        # location loss
        location_loss = self.location_loss(loc, size, angle, loc0, size0, angle0)
        
        # focal loss
        focal_loss = self.focal_loss(clf, clf0)

        # direction loss
        direction_loss = self.direction_loss(heading, heading0)

        loss_sum = self.loc_weight * location_loss + self.focal_weight * focal_loss + \
                   self.heading_weight * direction_loss

        
        return loss_sum
    

    def location_loss(self, loc, size, angle, loc0, size0, angle0):
        """
        location loss
        """
        da = torch.sqrt(size[...,0]**2 + size[...,1]**2)
        ha = size[...,2]
        
        delta_x = (loc0[...,0] - loc[...,0])/da
        delta_y = (loc0[...,1] - loc[...,1])/da
        delta_z = (loc0[...,2] - loc[...,2])/ha

        delta_w = torch.log(self.relu(size0[...,0]/size[...,0])+0.0001)
        delta_l = torch.log(self.relu(size0[...,1]/size[...,1])+0.0001)
        delta_h = torch.log(self.relu(size0[...,2]/size[...,2])+0.0001)

        delta_theta =  torch.sin(angle0[...,:] - angle[...,:])
        
        zeros = torch.zeros(delta_x.shape)

        location_loss = self.smoothL1(delta_x,zeros) + self.smoothL1(delta_y,zeros) + self.smoothL1(delta_z,zeros) + \
                 self.smoothL1(delta_w,zeros) + self.smoothL1(delta_l,zeros) + self.smoothL1(delta_h,zeros) + self.smoothL1(delta_theta,zeros)
        
        return location_loss



    def focal_loss(self, clf, clf0):
        """
        focal loss
        input shape: batch_size*252*252*4*4
        """
        alpha = self.alpha
        gamma = self.gamma

        prob = torch.sigmoid(clf)
        pt = prob * clf0 + (1 - prob) * (1 - clf0)

        alpha_factor = alpha * clf0 + (1 - alpha) * (1 - clf0)
        focal_weight = alpha_factor * (1 - pt).pow(gamma)
        bce = self.bce_logits(clf, clf0)
        return (focal_weight * bce).sum()
                

    def direction_loss(self, heading, heading0):
        """
        Input shape: batch_size*252*252*4
        """
        return self.bce_heading(heading[...,:], heading0[...,:])



if __name__=='__main__':
    mode = 'test all'
    if mode == 'test all':
        ls = point_pillars_loss()
        loc1 = torch.randn([4,252,252,4,3])
        loc0 = torch.randn([4,252,252,4,3])

        size1 = torch.randn([4,252,252,4,3])
        size0 = torch.randn([4,252,252,4,3])

        angle1 = torch.randn([4,252,252,4])
        angle0 = torch.randn([4,252,252,4])

        clf0 = torch.randn([4,252, 252, 4, 4])
        clf1 = torch.randn([4, 252, 252, 4, 4])
        
        heading1 = torch.randn([4,252,252,4])
        heading0 = torch.randn([4,252,252,4])
        result = ls(loc=loc1, size=size1, angle=angle1, clf=clf1, heading=heading1,
                    loc0=loc0, size0=size0, angle0=angle0, clf0=clf0, heading0=heading0)
        print(result)
    elif mode == 'test focal loss':
        ls = point_pillars_loss()
        clf = torch.ones(4,252,252,4,4)
        clf0 = torch.ones(4,252,252,4,4)
        res = ls.focal_loss(clf, clf0)
        print(res)
    elif mode == 'test direction loss':
        ls = point_pillars_loss()
        heading = torch.ones(4,252,252,4)
        heading0 = torch.ones(4,252,252,4)
        res = ls.direction_loss(heading, heading0)
        print(res)


