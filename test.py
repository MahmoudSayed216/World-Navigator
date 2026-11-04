# # # from logger import logger
# # # # from logger import Logger


# # # # logger = Logger(debug_mode=True, logs_folder_path="logs")

# # # # logger.log("HELLO WOLRD1!", end="")
# # # # logger.log("HELLO WOLRD2!", end="")
# # # # logger.log("HELLO WOLRD3!", end="")


# # # # logger.checkpoint("HELLO WOLRD1!", end="")
# # # # logger.checkpoint("HELLO WOLRD2!", end="")
# # # # logger.checkpoint("HELLO WOLRD3!", end="")


# # # # logger.debug("HELLO WOLRD1!", end="")
# # # # logger.debug("HELLO WOLRD2!", end="")
# # # # logger.debug("HELLO WOLRD3!", end="")


# # # # logger.log("HELLO WOLRD1!")
# # # # logger.log("HELLO WOLRD2!")
# # # # logger.log("HELLO WOLRD3!")


# # # # logger.checkpoint("HELLO WOLRD1!")
# # # # logger.checkpoint("HELLO WOLRD2!")
# # # # logger.checkpoint("HELLO WOLRD3!")


# # # # logger.debug("HELLO WOLRD1!")
# # # # logger.debug("HELLO WOLRD2!")
# # # # logger.debug("HELLO WOLRD3!")



# # # # # # s = "{k1}, {k2}"
# # # # # # s = s.format(k1="3")
# # # # # # print(s)
# # # # # from string import Template
# # # # # template = Template(template="$color_code[$source:$counter]")
# # # # # template = template.safe_substitute(color_code = "red")
# # # # # # template = template.safe_substitute(source = "nigga")
# # # # # # template = template.safe_substitute(counter= "nigggga")
# # # # # print(template)

# # # logger.set_debug_mode(True)
# # # logger.set_logs_dir("logs")

# # # def log_stuff():
# # #     logger.log(f"""training started with paramter set:
# # #                     Model Name: ding ding
# # #                     xyz
# # #                     nigga nigger niggeroski""", end="")
# # # log_stuff()
# # # log_stuff()



# # x=  [1,2,3]
# # print(x[:2])
# # x.extend([0]*6)
# # print(x)


# x = list(range(20))

# x1 = x[:5]
# x2 = x[5:10]
# x3 = x[10:-1]

# print(x1)
# print(x2)
# print(x3)









import torch


t = torch.tensor([[1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6]])
print(t[:, [1,2]])












