from loss import *
from model import *
from utils import *

Tensor = torch.tensor


class Optimizing:
    def __init__(self, device, alpha=-2, beta=9, iterations=500, lr=1):
        self.STYLE_WEIGHT = beta * np.array([1.50, 0.80, 0.25, 0.25, 0.25]) # what if make this vary? the later, the haigher for the deeper
        self.CONTENT_WEIGHT = alpha * np.array([1])
        self.LEARNING_RATE = lr
        self.CONTENT_PATH = 'images/content.jpg'
        self.STYLE_PATH   = 'images/style.jpg'
        self.RESULT_PATH  = 'result.jpg'
        self.MAX_ITERATIONS = iterations
        self.SHOW_ITERATIONS = 50
        self.DEVICE = device

    def load_images(self, from_content=True):
        content_image = load_img(self.CONTENT_PATH).to(self.DEVICE)
        style_image   = load_img(self.STYLE_PATH).to(self.DEVICE)
        if from_content:
            result_image = content_image.clone()
        result_image.requires_grad = True

        return content_image, style_image, result_image

    def save_result(self):
        show_img(self.result_image).save(self.RESULT_PATH)

    def style_transfer(self):
        model = NSTModel()
        model.to(self.DEVICE)
        content_image, style_image, result_image = self.load_images()
        optim = torch.optim.LBFGS([result_image], lr=self.LEARNING_RATE)

        style_targets   = [GramMatrix()(A).detach() for A in model(style_image)[0]]
        content_targets = [A.detach() for A in model(content_image)[1]]

        n_iterations = [0] # avoid error due to locality
        while n_iterations[0] <= self.MAX_ITERATIONS:
            def closure():
                optim.zero_grad()
                style_outputs, content_outputs = model(result_image)
                content_loss = [self.CONTENT_WEIGHT[i] * ContentLoss()(Y, content_targets[i]) for i, Y in enumerate(content_outputs)]
                style_loss   = [self.STYLE_WEIGHT[i]   * GramMSELoss()(Y, style_targets[i]) for i, Y in enumerate(style_outputs)]

                loss = sum(content_loss + style_loss)
                loss.backward()

                n_iterations[0] += 1
                if n_iterations[0] % self.SHOW_ITERATIONS == (self.SHOW_ITERATIONS - 1):
                    print('Iteration: %d, Style Loss: %.4f, Content Loss: %.4f' % (n_iterations[0], sum(style_loss).item(), sum(content_loss).item()))
                return loss

            optim.step(closure)
        self.result_image = result_image
        return result_image
