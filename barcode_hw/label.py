import qrcode
from barcode import Code128, EAN13, EAN14
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
import datetime

writer = ImageWriter()
writer.set_options({
    "module_width": 0.08,
    "module_height": 20.0,
    "quiet_zone": 20.0,
    "text_distance": 2.0,
    "write_text": False,
})

def gen_barcode(val, fname="barcode"):
    return Code128(val, writer=writer).save(fname)

def calc_check(num):
    total = 0
    for i, d in enumerate(num[::-1]):
        n = int(d)
        total += n * 3 if i % 2 == 0 else n
    return str((10 - (total % 10)) % 10)

def add_text(bfile, txt, font, cover=80, extra=50):
    im = Image.open(bfile).convert("RGB")
    w, h = im.size
    crop = im.crop((0,0,w,h-cover))
    new_h = crop.height + extra
    new = Image.new("RGB", (w,new_h), "white")
    new.paste(crop, (0,0))
    d = ImageDraw.Draw(new)
    bx = d.textbbox((0,0), txt, font=font)
    tw, th = bx[2]-bx[0], bx[3]-bx[1]
    x = (w-tw)//2
    y = crop.height + (extra-th)//2
    d.text((x,y), txt, font=font, fill="black")
    return new

def gen_qr(data, size=220): # QRcode
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=10, border=2)
    qr.add_data(data); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size,size), Image.LANCZOS)

def paste_bar_vert_no_label(canvas, im, x, y, width, scale=0.8):
    ow, oh = im.size
    ratio = width / ow
    nw = int(ow * ratio)
    nh = int(oh * ratio * scale)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas.paste(im2, (x,y))
    return y + nh + 20



if __name__ == "__main__":
    W, H = 1240, 1748  # A6
    canvas = Image.new("RGB", (W,H), "white")
    draw = ImageDraw.Draw(canvas)

    fpath = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    try:
        f_text = ImageFont.truetype(fpath, 38)
        f_sscc = ImageFont.truetype(fpath, 35)
    except:
        f_text = ImageFont.load_default()
        f_sscc = ImageFont.load_default()

    sid = input("请输入学号后4位：").strip()
    sup = "6907992" #伊利
    today = datetime.date.today()
    prod = today.strftime("%Y-%m-%d")
    exp = (today + datetime.timedelta(days=14)).strftime("%Y-%m-%d")
    from_txt = [
        "FROM",
        "供应商: 伊利股份有限公司",
        "地址: 内蒙古呼和浩特市敕勒川乳业开发区伊利大街1号",
        "厂区电话: 0471-3357805"
    ]
    to_txt = [
        "TO",
        "收货人: 联华超市",
        "到货地: 联华上海配送中心",
        "地址: 上海市静安区北宝兴路624号",
        "联系电话: 021-52785468"
    ]
    goods_txt = [
        "货物名称: 伊利金典鲜牛奶(250ml×12)",
        "数量: 125箱 1托盘",
        f"生产日期: {prod}",
        f"保质期: {exp}"
    ]

    gtin_base = sup + "0"*(13-len(sup)-len(sid)) + sid
    gtin_file = EAN14(gtin_base, writer=ImageWriter()).save("gtin_barcode")
    gtin_im = Image.open("gtin_barcode.png").convert("RGB")
    seq_len = 17 - len(sup)
    seq = sid.zfill(seq_len) # 0
    base = sup + seq
    chk = calc_check(base)
    sscc_val = "(00)" + base + chk
    sscc_file = gen_barcode(sscc_val, "sscc_barcode")
    sscc_im = add_text(sscc_file, sscc_val, f_sscc)
    qr1 = gen_qr("https://surl.amap.com/3z8Q7SQg08b", 200)
    qr2 = gen_qr("https://re.jd.com/search?keyword=金典鲜牛奶&enc=utf-8", 150)
    pad = 20
    gap = 46

    # FROM
    x0, y0, w0, h0 = 50, 50, 1000, 200
    draw.rectangle([x0,y0,x0+w0,y0+h0], outline="black", width=3)
    ty = y0 + pad
    for l in from_txt:
        draw.text((x0+pad, ty), l, font=f_text, fill="black")
        ty += gap
    y_from = y0 + h0

    # TO
    x1, y1, w1, h1 = 50, y_from+30, 1000, 300
    draw.rectangle([x1,y1,x1+w1,y1+h1], outline="black", width=3)
    ty = y1 + pad
    for l in to_txt:
        draw.text((x1+pad, ty), l, font=f_text, fill="black")
        ty += gap
    canvas.paste(qr1, (x1+w1-200-pad, y1+pad))
    y_to = y1 + h1

    # 牛奶
    y_cur = max(y_from, y_to) + 30
    x2, y2, w2, h2 = 50, y_cur, W-100, 300
    draw.rectangle([x2,y2,x2+w2,y2+h2], outline="black", width=3)
    ty = y2 + pad
    for l in goods_txt:
        draw.text((x2+pad, ty), l, font=f_text, fill="black")
        ty += gap
    canvas.paste(qr2, (x2+w2-150-pad, y2+(h2-150)//2))
    y_goods = y2 + h2

    # 条码
    y_cur = y_goods + 30
    x3, y3, w3, h3 = 50, y_cur, W-100, 800
    draw.rectangle([x3,y3,x3+w3,y3+h3], outline="black", width=3)
    y_next = paste_bar_vert_no_label(canvas, gtin_im, x3+pad, y3+100, int(W*2/3), 0.5)
    paste_bar_vert_no_label(canvas, sscc_im, x3+pad, y_next+30, int(W*3/4), 0.5)
    canvas.save("test_label.png")


