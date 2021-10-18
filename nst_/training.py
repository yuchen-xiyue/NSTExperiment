from loss import *
from model import *
from utils import *

Tensor = torch.tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'


STYLE_WEIGHT = np.array([1.50, 0.80, 0.25, 0.25, 0.25]) # what if make this vary? the later, the haigher for the deeper
CONTENT_WEIGHT = np.array([1])
LEARNING_RATE = 1
CONTENT_PATH = 'images/content.jpg'
STYLE_PATH   = 'images/style.jpg'
RESULT_PATH  = 'result.jpg'

MAX_ITERATIONS = 500
SHOW_ITERATIONS = 50

content_image = load_img(CONTENT_PATH).to(device)
style_image   = load_img(STYLE_PATH).to(device)
result_image  = content_image.clone()

result_image.requires_grad = True


def style_transfer():
    model = NSTModel()
    model.to(device)
    optim = torch.optim.LBFGS([result_image], lr=LEARNING_RATE)

    style_targets   = [GramMatrix()(A).detach() for A in model(style_image)[0]]
    content_targets = [A.detach() for A in model(content_image)[1]]

    n_iterations = [0] # avoid error due to locality
    while n_iterations[0] <= MAX_ITERATIONS:
        def closure():
            optim.zero_grad()
            style_outputs, content_outputs = model(result_image)
            content_loss = [CONTENT_WEIGHT[i] * ContentLoss()(Y, content_targets[i]) for i, Y in enumerate(content_outputs)]
            style_loss   = [STYLE_WEIGHT[i] * GramMSELoss()(Y, style_targets[i]) for i, Y in enumerate(style_outputs)]

            loss = sum(content_loss + style_loss)
            loss.backward()

            n_iterations[0] += 1
            if n_iterations[0] % SHOW_ITERATIONS == (SHOW_ITERATIONS - 1):
                print('Iteration: %d, Style Loss: %.4f, Content Loss: %.4f' % (n_iterations[0], sum(style_loss).item(), sum(content_loss).item()))
            return loss

        optim.step(closure)
    return result_image